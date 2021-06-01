import base64
import os
import json
import requests
from io import BytesIO
import pandas as pd
import boto3
from weasyprint import CSS, HTML
from weasyprint.formatting_structure.boxes import InlineReplacedBox


def _get_boto_session():
    session = boto3.Session(region_name=os.environ["AWS_REGION"],
                            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
                            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"])
    return session


def _write_object_to_s3(bucket: str, key: str, buffer: BytesIO):
    """ Writes an object to S3 """
    s3_client = _get_boto_session().client("s3")
    s3_client.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())


def write_json_to_s3(json_dict, bucket: str, key: str):
    """ Write a Python Dict to S3 as JSON """
    buffer = BytesIO()
    buffer.write(json.dumps(json_dict).encode("utf-8"))
    buffer.seek(0)
    _write_object_to_s3(bucket=bucket, key=key, buffer=buffer)


def write_df_to_s3(dataframe: pd.DataFrame,
                   bucket: str,
                   key: str,
                   outputformat: str = "parquet"):
    """ Write a Pandas dataframe to S3 as Parquet """
    buffer = BytesIO()
    if outputformat == "parquet":
        dataframe.to_parquet(buffer, engine="pyarrow", index=False)
    elif outputformat == "csv":
        dataframe.to_csv(buffer, index=False)
    else:
        raise Exception("Unknown format")
    _write_object_to_s3(bucket=bucket, key=key, buffer=buffer)


def streamlit_report_to_pdf(title,
                            dataset_name,
                            figures,
                            descriptions,
                            write_html=True,
                            out_folder="./reports"):
    template = (
        ""
        "<figure>"
        '<img style="width:{width};height:{height}" src="data:image/svg+xml;base64,{image}">'
        "<br>"
        "<figcaption>{caption}</figcaption>"
        "</figure>"
        ""
    )
    images = [
        base64.b64encode(
            figure.to_image(
                format="svg", width=figure.layout["width"], height=figure.layout["height"], scale=0.8, engine="kaleido"
            )
        ).decode("utf-8")
        for figure in figures
    ]

    report_html = f"<h1>{title}</h1>"
    for i, image in enumerate(images):
        img = template.format(
            image=image, caption=f"Description:\n{descriptions[i]}", width=figures[i].layout["width"], height=figures[i].layout["height"]
        )
        report_html += img

    if write_html:
        with open(f"{out_folder}/{dataset_name}_{'_'.join(title.lower().split(' '))}_report.html", "w") as f:
            f.write(report_html)

    htmldoc = HTML(string=report_html, base_url="")
    css = CSS(
        string="""
        body {
            font-family: "IBM Plex Sans", "Montserrat", "arial", "sans-serif";
            font-size: 14px;
            color: rgb(46,63,92)
        }
        @media print {
        a::after {
            content: " (" attr(href) ") ";
        }
        pre {
            white-space: pre-wrap;
        }
        @page {
            margin: 20px 20px 50px 20px;
            size: A4;
            @bottom-right {
                content: counter(page);
            }
            @bottom-center {
                content: url("file:///Users/ricomeinl/Desktop/talus/talus_data_analysis/img/talus_logo.png");
            }
        }""")
    pdf_doc = htmldoc.render(stylesheets=[css])

    for page in pdf_doc.pages:
        for children in page._page_box.descendants():
            if isinstance(children, InlineReplacedBox):
                needed_width = children.width + \
                    page._page_box.margin_left + \
                    page._page_box.margin_right

                # Override width only if the table doesn't fit
                if page.width < needed_width:
                    page._page_box.width = needed_width
                    page.width = needed_width

    pdf_doc.write_pdf(f"{out_folder}/{dataset_name}_{'_'.join(title.lower().split(' '))}_report.pdf")