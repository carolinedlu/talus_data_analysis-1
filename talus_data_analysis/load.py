import os
import json
import requests
from io import BytesIO
import pandas as pd
import boto3
from botocore.exceptions import ClientError
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from hurry.filesize import size


def _get_boto_session():
    session = boto3.Session(region_name=os.environ["AWS_REGION"],
                            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
                            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"])
    return session


def _read_object_from_s3(bucket: str, key: str) -> BytesIO:
    """ Reads an object from S3 """
    s3_resource = _get_boto_session().resource("s3")
    bucket = s3_resource.Bucket(bucket)
    data = BytesIO()
    bucket.download_fileobj(key, data)
    data.seek(0)
    return data


def read_df_from_s3(bucket: str,
                    key: str,
                    inputformat: str = None,
                    **df_kwargs) -> pd.DataFrame:
    """ Reads a Pandas dataframe from S3 """
    data = _read_object_from_s3(bucket=bucket, key=key)
    if inputformat == "parquet":
        return pd.read_parquet(data, **df_kwargs)
    elif inputformat == "txt":
        return pd.read_table(data, **df_kwargs)
    else:
        return pd.read_csv(data, **df_kwargs)
    return pd.DataFrame()


def read_json_from_s3(bucket: str, key: str):
    file_content = _read_object_from_s3(bucket=bucket, key=key)
    return json.loads(file_content.read())


def get_file_keys_in_bucket(bucket: str, key: str, file_type: str = ""):
    """ Gets all the file keys in a given bucket, return empty list if none exist """
    s3_client = _get_boto_session().client("s3")
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=key)
    contents = response.get("Contents", [])

    return [obj.get("Key") for obj in contents if _is_file(obj["Key"]) and obj["Key"].endswith(file_type)]


def _is_file(filepath):
    """ Checks whether the file has a file extension, return True if it does """
    return True if os.path.splitext(filepath)[1] else False


def file_exists_in_bucket(bucket: str, key: str):
    """ Checks whether a file key exists in bucket """
    s3_client = _get_boto_session().client("s3")
    try:
        file = s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        pass
    return False


def get_file_size(bucket: str, key: str, **kwargs):
    """ Gets the size for a file with key in given bucket """
    s3_client = _get_boto_session().client("s3")
    try:
        file = s3_client.head_object(Bucket=bucket, Key=key)
        return size(file["ContentLength"], **kwargs)
    except ClientError:
        pass
    return


def get_s3_file_sizes(bucket, keys, file_type):
    size_dicts = []
    for key in keys:
        file_keys = get_file_keys_in_bucket(bucket=bucket, key=key, file_type=file_type)

        for file_key in file_keys:
            file_size = get_file_size(bucket=bucket, key=file_key)
            size_dicts.append({"File": os.path.basename(file_key), "Type": file_key.split("/")[0], "Size": file_size})

    return pd.DataFrame(size_dicts).sort_values(by="File").reset_index(drop=True)


def _read_object_from_gdrive(gauth, key: str, **kwargs):
    """
    """
    # prevents error
    os.environ["DISPLAY"] = ""

    drive = GoogleDrive(gauth)

    fileList = drive.ListFile({"corpora": "drive",
                               "driveId": "0APHFhmQ_lBlpUk9PVA",
                               "includeItemsFromAllDrives": True,
                               "supportsAllDrives": True,
                               "q": f"title contains '{key}'"}).GetList()

    res = requests.get(fileList[0]["downloadUrl"], headers={"Authorization": "Bearer " + gauth.attr['credentials'].access_token})
    data = BytesIO(res.content)
    return data


def read_excel_from_gdrive(gauth,
                           key: str,
                           sheet: str,
                           remove_unnamed: bool = False,
                            **kwargs) -> pd.DataFrame:
    """ Reads an Excel Sheet into a DataFrame """
    data = _read_object_from_gdrive(gauth=gauth, key=key)
    df = pd.read_excel(data, usecols=None, sheet_name=sheet, **kwargs)

    if remove_unnamed:
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    return df


def read_csv_from_gdrive(gauth,
                         key: str,
                         **kwargs) -> pd.DataFrame:
    """ Reads a csv file into a DataFrame """
    data = _read_object_from_gdrive(gauth=gauth, key=key)

    return pd.read_csv(data, usecols=None, **kwargs)