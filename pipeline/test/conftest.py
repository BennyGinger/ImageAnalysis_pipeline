import pytest

@pytest.fixture(scope='session')
def test_image_path_nd2_1():
    return '/home/Test_images/nd2/Run1/c1z25t25v1_nd2.nd2'

@pytest.fixture(scope='session')
def test_image_path_nd2_2():
    return '/home/Test_images/nd2/Run2/c2z25t23v1_nd2.nd2'

@pytest.fixture(scope='session')
def test_image_path_nd2_3():
    return '/home/Test_images/nd2/Run3/c3z1t1v3.nd2'

@pytest.fixture(scope='session')
def test_image_path_nd2_4():
    return '/home/Test_images/nd2/Run4/c4z1t91v1.nd2'

@pytest.fixture(scope='session')
def test_image_path_tif_1():
    return '/home/Test_images/tif/Run1/c1z25t25v1_tif.tif'

@pytest.fixture(scope='session')
def test_image_path_tif_2():
    return '/home/Test_images/tif/Run2/c2z25t23v1_tif.tif'

@pytest.fixture(scope='session')
def test_image_path_tif_4():
    return '/home/Test_images/tif/Run4/c4z1t91v1.tif'