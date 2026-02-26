import numpy as np
def load():
    Xs, ys = [], []
    for i in range(num_files):
        for j,source in enumerate(SOURCES):
            try:
                url = urls[source][i]
                filename = url.split('/')[-1].split('?')[0]

                fpath = _get_filepath(filename, url, cache_dir, file_hash=hashes['sha256'][i])

                # we succeeded, so don't continue trying to download this file
                break

            except Exception as e:
                print(str(e))

                # if this was our last source, raise an error
                if j == len(SOURCES) - 1:
                    m = 'Failed to download {} from any source.'.format(filename)
                    raise RuntimeError(m)

                # otherwise indicate we're trying again
                else:
                    print("Failed to download {} from source '{}', trying next source...".format(filename, source))

        # load file and append arrays
        with np.load(fpath) as f:
            Xs.append(f['X'])
            ys.append(f['y'])

    # get X array
    if pad:
        max_len_axis1 = max([X.shape[1] for X in Xs])
        X = np.vstack([_pad_events_axis1(x[...,:ncol], max_len_axis1) for x in Xs])
    else:
        X = np.asarray([x[x[:,0]>0,:ncol] for X in Xs for x in X], dtype='O')

    # get y array
    y = np.concatenate(ys)

    # chop down to specified amount of data
    if num_data > -1:
        X, y = X[:num_data], y[:num_data]

    return X, y