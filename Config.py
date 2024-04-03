class Config:
    def __init__(self, **kwargs) -> None:
        for name, value in kwargs.items():
            setattr(self, name, value)

    def save(self, file_path: str):
        raise NotImplementedError("Save function not implemented")
