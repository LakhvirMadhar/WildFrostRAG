"""Bling domain models: enemy bling drops and shop price listings.

Parsing the Bling, shop, and Clunkers wiki pages lives in
data_processing/pages/bling/parser.py.
"""

from pydantic import BaseModel, Field


class EnemyBlingDrop(BaseModel):
    """An enemy's base bling drop value."""

    card_name: str = Field(min_length=1)
    amount: int = Field(ge=0)


class ShopListing(BaseModel):
    """An item or charm listing in a shop."""

    card_name: str = Field(min_length=1)
    base_price: int = Field(ge=0)
