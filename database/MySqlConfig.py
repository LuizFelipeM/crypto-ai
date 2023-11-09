class MySqlConfig:
    user: str
    password: str
    raw_connection_string: str

    def __init__(self, user, password, raw_connection_string) -> None:
        self.user = user
        self.password = password
        self.raw_connection_string = raw_connection_string

    def connection_string(self) -> str:
        return self.raw_connection_string.replace("<User>", self.user).replace(
            "<Password>", self.password
        )
