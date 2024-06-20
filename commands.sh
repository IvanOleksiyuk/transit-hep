# Test with 10 samples
python mytoolkit/scripts/create_dummy_data.py --type spherical_gausssian --output /home/users/o/oleksiyu/WORK/hyperproject/dummydata/data.h5 --plot --n_samples 10 --n_features 2 --seed 42

# Small dataset for debugging purposes 10K
python mytoolkit/scripts/create_dummy_data.py --type spherical_gaussian --output dummydata/gauss_sph_2_10K.h5 --plot --n_samples 10000 --n_features 2 --seed 42
python mytoolkit/scripts/create_dummy_data.py --type gaussian_with_covariance --output dummydata/gauss_corr_2_10K.h5 --plot --n_samples 10000 --n_features 2 --seed 42

# Generate quadratic depencence dataset
python mytoolkit/scripts/create_dummy_data.py --type powers --output dummydata/powers_5_1M.h5 --plot --n_samples 1000000 --n_features 5 --seed 42

# CONCEPT
python mytoolkit/scripts/create_dummy_data.py --type spherical_gaussian --output /home/users/o/oleksiyu/WORK/hyperproject/dummydata/gauss_sph_4_10K.h5 --plot --n_samples 10000 --n_features 4 --seed 42
python mytoolkit/scripts/create_dummy_data.py --type gaussian_with_covariance --output /home/users/o/oleksiyu/WORK/hyperproject/dummydata/gauss_corr_4_10K.h5 --plot --n_samples 10000 --n_features 4 --seed 42

python mytoolkit/scripts/create_dummy_data.py --type gaussian_with_covariance --output /home/users/o/oleksiyu/WORK/hyperproject/dummydata/gauss_corr_4_1M.h5 --plot --n_samples 1000000 --n_features 4 --seed 42
python mytoolkit/scripts/create_dummy_data.py --type spherical_gausssian --output /home/users/o/oleksiyu/WORK/hyperproject/dummydata/gauss_sph_4_1M.h5 --plot --n_samples 1000000 --n_features 4 --seed 42

python mytoolkit/scripts/create_dummy_data.py --type gaussian_with_covariance --output /home/users/o/oleksiyu/WORK/hyperproject/dummydata/gauss_corr_2.h5 --plot --n_samples 1000000 --n_features 2 --seed 42

python mytoolkit/scripts/create_dummy_data.py --type gaussian_with_covariance --output /home/users/o/oleksiyu/WORK/hyperproject/dummydata/gauss_corr_4.h5 --plot --n_samples 1000000 --n_features 4 --seed 42

python mytoolkit/scripts/create_dummy_data.py --type gaussian_with_covariance --output /home/users/o/oleksiyu/WORK/hyperproject/dummydata/gauss_corr_8.h5 --plot --n_samples 1000000 --n_features 8 --seed 42

