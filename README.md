# Use Google Vision API to extract Text Annotations

## Description

Here Script use `Google Vision api` to extract `Text Annotations` in images.

## Requirement

* Python 3.x
* Credentials

## Setup

To install necessary library, simply use pip:

```bash
pip install google-cloud-vision
```

or,

```bash
pip install -r requirements.txt
```

Next, set up to authenticate with the Cloud Vision API using your project's service account credentials. See the [Vision API Client Libraries](https://cloud.google.com/vision/docs/libraries) for more information. Then, set the GOOGLE_APPLICATION_CREDENTIALS environment variable to point to your downloaded service account credentials:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/credentials-key.json
```

## Quick Start: Running script

Text Detection

```bash
$ python vision.py --images path/to/folder/contain/images 
```

# Result
## Words Annotations
![Words Annotations](https://github.com/docongminh/Simple-google-vision-api/blob/master/image_test/word_rect_test.jpg)

## Character Annotations
![Characters Annotations](https://github.com/docongminh/Simple-google-vision-api/blob/master/image_test/char_rect_test.jpg)

## Image Sizing suggestion

To enable accurate image detection within the Google Cloud Vision API, images should generally be a minimum of 640 x 480 pixels (about 300k pixels). Full details for different types of Vision API Feature requests are shown below:

| Vision API Feature | Recommended Size | Notes |
|---|---|---|
| FACE_DETECTION | 1600 x 1200 | Distance between eyes is most important |
| LANDMARK_DETECTION | 640 x 480 |   |
| LOGO_DETECTION | 640 x 480 |   |
| LABEL_DETECTION | 640 x 480 |   |
| TEXT_DETECTION | 1024 x 768 | OCR requires more resolution to detect characters |
| SAFE_SEARCH_DETECTION | 640 x 480 |   |
