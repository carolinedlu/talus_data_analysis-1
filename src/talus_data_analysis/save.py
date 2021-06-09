import base64
import json
import os

from io import BytesIO

import boto3
import pandas as pd

from fpdf import FPDF


def _get_boto_session():
    session = boto3.Session(
        region_name=os.environ["AWS_REGION"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )
    return session


def _write_object_to_s3(bucket: str, key: str, buffer: BytesIO):
    """Writes an object to S3"""
    s3_client = _get_boto_session().client("s3")
    s3_client.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())


def write_json_to_s3(json_dict, bucket: str, key: str):
    """Write a Python Dict to S3 as JSON"""
    buffer = BytesIO()
    buffer.write(json.dumps(json_dict).encode("utf-8"))
    buffer.seek(0)
    _write_object_to_s3(bucket=bucket, key=key, buffer=buffer)


def write_df_to_s3(
    dataframe: pd.DataFrame, bucket: str, key: str, outputformat: str = "parquet"
):
    """Write a Pandas dataframe to S3 as Parquet"""
    buffer = BytesIO()
    if outputformat == "parquet":
        dataframe.to_parquet(buffer, engine="pyarrow", index=False)
    elif outputformat == "csv":
        dataframe.to_csv(buffer, index=False)
    else:
        raise Exception("Unknown format")
    _write_object_to_s3(bucket=bucket, key=key, buffer=buffer)


class PDF(FPDF):
    def header(self):
        self.set_font(family="Helvetica", style="B", size=15)
        self.set_fill_color(255, 255, 255)
        w = self.get_string_width(self.title) + 6
        self.set_x((210 - w) / 2)
        self.cell(w, 9, self.title, 0, 1, "L", 1)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font(family="Helvetica", style="I", size=8)
        self.set_text_color(128)
        self.cell(0, 10, "Page " + str(self.page_no()), 0, 0, "C")

    def chapter_title(self, num, label):
        self.set_font(family="Helvetica", style="B", size=12)
        self.set_fill_color(200, 220, 255)
        self.multi_cell(w=0, h=6, txt="%d: %s" % (num, label), border=0, align="L")
        self.ln(4)

    def chapter_body(self, name, description, width=210, height=297):
        self.set_font(family="Helvetica", style="", size=12)
        self.multi_cell(w=0, h=5, txt=description)
        self.image(name, w=width)
        self.ln()

    def print_chapter(self, num, title, name, description, width=210, height=None):
        self.add_page()
        self.chapter_title(num, title)
        self.chapter_body(
            name=name, description=description, width=int(width * 0.8), height=height
        )


def create_download_link(val, filename):
    b64 = base64.b64encode(val)
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}">Download file</a>'


def streamlit_report_to_pdf(out_filename, title, figures, titles, descriptions):
    figurepaths = []
    for i, figure in enumerate(figures):
        figurepath = f"/tmp/fig{i+1}.png"
        figure.write_image(figurepath)
        figurepaths.append(figurepath)

    pdf = PDF()
    pdf.set_title(title)
    for i, path in enumerate(figurepaths):
        pdf.print_chapter(
            num=i + 1, title=titles[i], name=path, description=descriptions[i]
        )

    pdf_output = pdf.output(out_filename, dest="S")

    return pdf_output
