from src.DateTimeHelper import DateTimeHelper
from datetime import datetime, timedelta, timezone

class ContextCacheObj():
    
    def __init__(self, id, data):
        """
        Initialize cache object.
        args:
            id (int): ID of object.
            data (any): Object to cache.
        """
        self._id = id
        self.expiration = DateTimeHelper.get_current_datetime() + timedelta(days=2)
        self._data = data

    @property
    def id(self):
        """
        Get ID of object.
        
        returns (int): Object ID.
        """
        return self._id
    
    @property
    def data(self):
        """
        Get data from of object.

        args:
            obj_id (int): Object data.
        """
        return self._data