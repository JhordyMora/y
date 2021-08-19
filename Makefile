# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* TFM_TrainAtScalePipeline/*.py

black:
	@black scripts/* TFM_TrainAtScalePipeline/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

install:
	@pip install . -U

all: clean install test black check_code

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
PYPI_USERNAME=<AUTHOR>
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)

# ----------------------------------
#      GCP
# ----------------------------------
### GCP configuration - - - - - - - - - - - - - - - - - - -

### GCP Project - - - - - - - - - - - - - - - - - - - - - -

# not required here

### GCP Storage - - - - - - - - - - - - - - - - - - - - - -

BUCKET_NAME = "wagon-data-672-mora-tabares"

##### Data  - - - - - - - - - - - - - - - - - - - - - - - -

# train data file location
# /!\ here you need to decide if you are going to train using the provided and uploaded data/train_1k.csv sample file
# or if you want to use the full dataset (you need need to upload it first of course)
BUCKET_TRAIN_DATA_PATH = 'data/train_1k.csv'

##### Training  - - - - - - - - - - - - - - - - - - - - - -

# will store the packages uploaded to GCP for the training
BUCKET_TRAINING_FOLDER = 'trainings'

##### Model - - - - - - - - - - - - - - - - - - - - - - - -

# model folder name (will contain the folders for all trained model versions)
MODEL_NAME = 'taxifarepipeline'

# model version folder name (where the trained model.joblib file will be stored)
MODEL_VERSION = 'v1'

#Storage location  (path and folders we want to created ) in GCP
STORAGE_LOCATION = 'models/taxifarepipeline/model.joblib'
### GCP AI Platform - - - - - - - - - - - - - - - - - - - -

##### Machine configuration - - - - - - - - - - - - - - - -

REGION = europe-west1

PYTHON_VERSION = 3.7
FRAMEWORK = scikit-learn
RUNTIME_VERSION = 1.15

##### Package params  - - - - - - - - - - - - - - - - - - -

PACKAGE_NAME = TFM_TrainAtScalePipeline#El nombre de este paquete
FILENAME = trainer

##### Job - - - - - - - - - - - - - - - - - - - - - - - - -
JOB_NAME=taxi_fare_training_pipeline_$(shell date +'%Y%m%d_%H%M%S')
### - - - - - - - - - - - - - - - - - - - - - - - - - - - -

STORAGE_LOCATION = 'models/taxifarepipeline/model.joblib'

run_locally:
	@python -m ${PACKAGE_NAME}.${FILENAME}

gcp_submit_training:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
		--job-dir gs://${BUCKET_NAME}/${BUCKET_TRAINING_FOLDER} \
		--package-path ${PACKAGE_NAME} \
		--module-name ${PACKAGE_NAME}.${FILENAME} \
		--python-version=${PYTHON_VERSION} \
		--runtime-version=${RUNTIME_VERSION} \
		--region ${REGION} \
		--stream-logs

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ __pycache__
	@rm -fr build dist *.dist-info *.egg-info
	@rm -fr */*.pyc
