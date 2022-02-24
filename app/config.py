class Base:
    """Base config class."""

    SWAGGER = {
        "title": "Analysis with Machine Learning App API",
        "uiversion": 3,
        "Description": "API Documentation for base app",
    }


class Development(Base):
    """Development config class."""

    ...


class Production(Base):
    """Production config class."""

    ...


class Testing(Base):
    """Testing config class."""

    ...


class Staging(Base):
    """Staging config class."""

    ...
