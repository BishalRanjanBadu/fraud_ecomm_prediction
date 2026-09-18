"""In-memory stand-in for the three S3 calls the service and scripts make."""
import io

from botocore.exceptions import ClientError


class FakeS3:
    def __init__(self, store=None):
        self.store = dict(store or {})
        self.puts = []

    def get_object(self, Bucket, Key):
        if Key not in self.store:
            raise ClientError({"Error": {"Code": "NoSuchKey", "Message": Key}}, "GetObject")
        return {"Body": io.BytesIO(self.store[Key])}

    def head_object(self, Bucket, Key):
        if Key not in self.store:
            raise ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject")
        return {}

    def put_object(self, Bucket, Key, Body, ContentType=None):
        self.store[Key] = Body
        self.puts.append(Key)
