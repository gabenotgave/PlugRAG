from src.DateTimeHelper import DateTimeHelper
import threading

class CacheService():
    
    def __init__(self):
        """
        Initialize cache and lock mechanism.
        """
        self._cache = {}
        self._lock = threading.RLock()

    def add(self, obj):
        """
        Add object to cache service.

        args:
            obj (ContextCacheObj): Object compatible with cache service.
        """
        with self._lock:
            self._cache[obj.id] = obj
    
    def get(self, obj_id):
        """
        Get object from cache via ID.

        args:
            obj_id (int): ID of object to retrieve.
        
        returns (ContextCacheObj): Cache object.
        """
        with self._lock:
            obj = self._cache.get(obj_id)
            if obj and obj.expiration > DateTimeHelper.get_current_datetime():
                return obj
            if obj:
                del self._cache[obj_id]
            return None
    
    def cleanup(self):
        """
        Clean up expired cache objects.
        """
        with self._lock:
            now = DateTimeHelper.get_current_datetime()
            self._cache = {k: v for k, v in self._cache.items() if v.expiration > now}