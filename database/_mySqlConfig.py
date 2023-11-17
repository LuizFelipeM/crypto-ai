class MySqlConfig:
    connection_string: str

    def __init__(self, connection_string: str) -> None:
        self.connection_string = connection_string
