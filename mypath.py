class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'tgs_salt_identification':
            return '/content/competition_data/train/images', '/content/competition_data/train/masks'  # folder that contains TGS.
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError