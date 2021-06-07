import sqlite3
import tempfile

import pandas as pd

from .load import _read_object_from_s3


class Elib:
    def __init__(self, key, bucket=None):
        """Initialize a new connection to a file by downloading it as a tmp file"""
        if not bucket:
            self.file_name = key
        else:
            elib = _read_object_from_s3(bucket=bucket, key=key)
            elib_content = elib.read()
            self.tmp = tempfile.NamedTemporaryFile()
            self.tmp.write(elib_content)
            self.file_name = self.tmp.name

        # connect to tmp file
        self.connection = sqlite3.connect(self.file_name)
        self.cursor = self.connection.cursor()

    def execute_sql(self, sql, use_pandas=False):
        """Execute a given SQL command"""
        if use_pandas:
            return pd.read_sql_query(sql=sql, con=self.connection)
        else:
            return self.cursor.execute(sql)

    def close(self):
        """Closes and removes the tmp file and the connection"""
        self.tmp.close()
