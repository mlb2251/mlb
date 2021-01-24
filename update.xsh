if input("did you updated the version number in setup.py? [y/n]").strip() != 'y':
    print("aborting...")
    exit(0)
python -m pip install --user --upgrade setuptools wheel
rm dist/*
python setup.py sdist bdist_wheel
python -m pip install --user --upgrade twine
python -m twine upload dist/*
