from setuptools import find_packages, setup

package_name = 'car_control_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'casadi'],
    zip_safe=True,
    maintainer='selena',
    maintainer_email='selenahuang0218@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "car_control_node = car_control_pkg.main:main"
        ],
    },
)
