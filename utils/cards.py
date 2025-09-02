import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from pathlib import Path
from bs4 import BeautifulSoup, Comment
import logging

logger = logging.getLogger(__name__)

class CardType(Enum):
    """
    What type is the card
    """
    LEADER = ('leaders', [])
    PETS = ('pets', ['companions'])  # pets are a subtype of companions
    COMPANIONS = ('companions', [])
    SHADES = ('shades', [])
    CLUNKERS = ('clunkers', [])
    ITEMS = ('items', [])
    ENEMIES = ('enemies', [])
    ENEMY_CLUNKERS = ('enemy_clunkers', ['enemies', 'clunkers'])  # inherits from both
    MINIBOSSES = ('minibosses', ['enemies'])  # minibosses are a subtype of enemies
    BOSSES = ('bosses', ['enemies'])  # bosses are a subtype of enemies
    

    def __new__(cls, value, parents=None):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.parents = parents or []
        return obj
    
    @property
    def has_parents(self) -> bool:
        """Check if this card type has parent types"""
        return len(self.parents) > 0



class TribeExclusivity(Enum):
    """
    Defins which tribe(s) a card belongs to
    """
    SNOWDWELLERS = 'Snowdwellers'
    SHADMANCERS = 'Shademancers'
    CLUNKMASTERS = 'Clunkmasters'
    ALL_TRIBES = 'All' # Card belongs to all three tribes


    @property
    def is_exclusive(self) -> bool:
        """Check if this card is exclusive to a single tribe"""
        return self != TribeExclusivity.ALL_TRIBES
    
    
    @property
    def is_universal(self) -> bool:
        return self == TribeExclusivity.ALL_TRIBES
    

    def get_tribes(self) -> list[str]:
        """
        Get list of tribe names this exclusivity represents
        """

        if self.is_universal:
            return [tribe.value for tribe in TribeExclusivity if tribe.is_exclusive]
        else:
            return [self.value]


@dataclass
class CardInfo:
    card_name: str
    card_type: CardType
    card_url: str
    card_html: Optional[str] = None

    # Card Stats
    card_description: Optional[str] = None
    health: Optional[int] = None
    attack: Optional[int] = None
    scrap: Optional[int] = None  # Alternative to Heatlh
    counter: Optional[int] = None
    other_stats: Optional[str] = None
    abilities_normalized: Optional[str] = None
    abilities_specific: Optional[str] = None
    
    # Tribe Exclusivity
    tribe_exclusivity: Optional[TribeExclusivity] = None


    def sanitized_name(self) -> str:
        """Get sanitized card name safe for filenames"""
        return re.sub(r'[\\/:*?"<>|]', '', self.card_name)


    def save_path(self) -> str:
        """Generate the save path for this card's HTML"""
        return  f'data/structured_outputs/{self.card_type.value}/{self.sanitized_name()}.html'


    def save_html(self) -> bool:
        """
        Save the card's HTML to file with proper directory creation and cleaning
        
        Returns:
            bool: True is saved sucessfully, False otherwise
        """

        if self.card_html is None:
            logger.warning(f"No HTML content to save for {self.card_name}")
            return False
        
        try:
            # Create directory if it doesn't exist
            save_path = Path(self.save_path())
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Get HTML file
            soup = BeautifulSoup(self.card_html, 'html.parser')

            # Remove comments in HTML
            comments = soup.find_all(string=lambda text: isinstance(text, Comment))
            for comment in comments:
                comment.extract()

            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(soup.prettify())

        except Exception as e:
            logger.error(f'Failed to save HTML for {self.card_name}: {e}')


    # This really is for the card_type, we'd have to do another function for the tribe exclusiviy
    def parse_html(self) -> bool:
        """
        Parse the HTML and populate the card stats fields
        
        Returns:
            bool: True if parsing succeeded, False otherwise
        """
        if not self.card_html:
            logger.warning(f"No HTML content to parse for {self.card_name}")
            return False
        
        try:
            soup = BeautifulSoup(self.card_html, 'html.parser')
            
            # Extract description
            description_tag = soup.find("meta", attrs={'name': 'description'})
            if description_tag:
                self.card_description = description_tag.get("content", "")
            
            # Extract stats from infobox
            infobox = soup.find('table', {'id': 'infobox'})
            if infobox:
                rows = infobox.find_all('tr')
                if len(rows) >= 4:
                    # Stats are in rows 2 and 3
                    stats_headers = [th.text.strip() for th in rows[2].find_all('th')]
                    stats_values = [td.text.strip() for td in rows[3].find_all('td')]
                    
                    stats = dict(zip(stats_headers, stats_values))
                    
                    # Dynamically populate matching fields
                    for stat_name, value in stats.items():
                        attr_name = stat_name.lower()
                        
                        if hasattr(self, attr_name):
                            if value.strip() == "":
                                setattr(self, attr_name, None)
                            elif value.isdigit():
                                setattr(self, attr_name, int(value))
                            else:
                                setattr(self, attr_name, value)
                    
                    # Look for "Other Stats" section (effects)
                    for i, row in enumerate(rows):
                        th = row.find('th')
                        if th and th.text.strip() == "Other Stats":
                            if i + 1 < len(rows):
                                other_stats_row = rows[i + 1]
                                td = other_stats_row.find('td')
                                if td:
                                    other_stats_text = td.get_text(strip=True)
                                    self.other_stats = other_stats_text if other_stats_text else None
                            break
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to parse HTML for {self.card_name}: {e}")
            return False
    

    def __str__(self) -> str:
        """String representation of the card based on to_dict"""
        card_data = self.to_dict()
        lines = [f"{key.replace('_', ' ').title()}: {value}" for key, value in card_data.items()]
        return "\n".join(lines)


    def to_dict(self) -> dict:
        """
        Dictionary representation excluding None values and card_html, for neo4j consumption.
        Handles CardType(Enum), which neo4j can't do on it's own
        """
        result = {}
        for k, v in self.__dict__.items():
            if v is None or k == "card_html":
                continue
            # Convert enum to its value
            if isinstance(v, Enum):
                result[k] = v.value
            else:
                result[k] = v
        return result

