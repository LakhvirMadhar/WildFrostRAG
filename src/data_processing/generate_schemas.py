import requests
from bs4 import BeautifulSoup


def generate_card_type_html_schema(schema_url: str='https://wildfrostwiki.com/index.php?title=Baby_Snowbo'):
    """
    Given a base schema url, take the cards, break them down into their schema, and save accordingly.

    schema_url (str): A base url to extract the card type schema. Default link is provided
    """
    response = requests.get(schema_url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')

    card_schema = soup.find('table', {'class': 'wikitable', 'id':'navbox'})

    card_data = {}

    rows = card_schema.find_all('tr')[1:]

    for row in rows:
        card_list = []
        card_type = row.find('th')
        card_type_data = row.find('td')
        card_names = card_type_data.find_all('a')

        category_name = card_type.get_text().strip().lower().replace(' ', '_')

        print(f"Schema Text: {category_name}")

        for n in card_names:
            card_name = n.get_text().strip()
            # print(f'{card_name}')
            card_list.append(card_name)
        
        card_data[category_name] = card_list

    return card_data