"""aiotcvectordb.exceptions

对外提供与 tcvectordb.exceptions 等价的异常类型，但模块归属在 aiotcvectordb，
便于异步 SDK 使用者统一从本包导入和感知。
"""

from tcvectordb.exceptions import (
    ErrorCode as ErrorCode,
    ERROR_MESSAGE_NETWORK_OR_AUTH as ERROR_MESSAGE_NETWORK_OR_AUTH,
    VectorDBException as _VectorDBException,
    ParamError as _ParamError,
    NoConnectError as _NoConnectError,
    ConnectError as _ConnectError,
    ServerInternalError as _ServerInternalError,
    DescribeCollectionException as _DescribeCollectionException,
    GrpcException as _GrpcException,
)


class VectorDBException(_VectorDBException):
    pass


class ParamError(_ParamError):
    pass


class NoConnectError(_NoConnectError):
    pass


class ConnectError(_ConnectError):
    pass


class ServerInternalError(_ServerInternalError):
    pass


class DescribeCollectionException(_DescribeCollectionException):
    pass


class GrpcException(_GrpcException):
    pass


__all__ = [
    "ErrorCode",
    "ERROR_MESSAGE_NETWORK_OR_AUTH",
    "VectorDBException",
    "ParamError",
    "NoConnectError",
    "ConnectError",
    "ServerInternalError",
    "DescribeCollectionException",
    "GrpcException",
]
