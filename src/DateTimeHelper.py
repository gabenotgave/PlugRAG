from datetime import datetime, timezone

class DateTimeHelper():

    def get_current_datetime():
        """
        Get current date and time in UTC.
        
        returns (datetime): Current date and time in UTC.
        """
        return datetime.now(timezone.utc)