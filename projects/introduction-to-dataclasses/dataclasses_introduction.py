# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Introduction to Dataclasses

# %%
from dataclasses import dataclass

@dataclass
class Position:
    name: str
    lon: float
    lat: float


# %%
pos = Position('Oslo', 10.8, 59.9)
print(pos)


# %%
pos.lat


# %%
print(f'{pos.name} is at {pos.lat}°N, {pos.lon}°E')


# %%
@dataclass
class Position:
    name: str
    lon: float = 0.0
    lat: float = 0.0


# %%
Position('Default Island')


# %%
Position('Greenwich', lat=51.8)


# %%
Position('Vancouver', -123.1, 49.3)


# %%
from dataclasses import dataclass
from typing import Any

@dataclass
class WithoutExplicitTypes:
    name: Any
    value: Any = 42


# %%
WithoutExplicitTypes('')


# %%
Position(3.14, 'pi day', 2018)


# %%
from math import asin, cos, radians, sin, sqrt

@dataclass
class Position:
    name: str
    lon: float = 0.0
    lat: float = 0.0

    def distance_to(self, other):
        r = 6371  # Earth radius in kilometers
        lam_1, lam_2 = radians(self.lon), radians(other.lon)
        phi_1, phi_2 = radians(self.lat), radians(other.lat)
        h = (sin((phi_2 - phi_1) / 2)**2
             + cos(phi_1) * cos(phi_2) * sin((lam_2 - lam_1) / 2)**2)
        return 2 * r * asin(sqrt(h))


# %%
oslo = Position('Oslo', 10.8, 59.9)
vancouver = Position('Vancouver', -123.1, 49.3)
oslo.distance_to(vancouver)

# %% [markdown]
# ## A deck of cards

# %%
from typing import List

@dataclass
class PlayingCard:
    rank: str
    suit: str

@dataclass
class Deck:
    cards: List[PlayingCard]


# %%
queen_of_hearts = PlayingCard('Q', 'Hearts')
ace_of_spades = PlayingCard('A', 'Spades')
two_cards = Deck([queen_of_hearts, ace_of_spades])

two_cards


# %%
RANKS = '2 3 4 5 6 7 8 9 10 J Q K A'.split()
SUITS = '♣ ♢ ♡ ♠'.split()

def make_french_deck():
    return [PlayingCard(r, s) for s in SUITS for r in RANKS]


# %%
make_french_deck()

# %% [markdown]
# How do we assign a default value to the deck?

# %%
#Incorrect method
from dataclasses import dataclass
from typing import List

@dataclass
class Deck:  # Will NOT work
    cards: List[PlayingCard] = make_french_deck()


# %%
#Correct method
from dataclasses import dataclass, field
from typing import List

@dataclass
class Deck:
    cards: List[PlayingCard] = field(default_factory=make_french_deck)


# %%
Deck()


# %%
@dataclass
class Position:
    name: str
    lon: float = field(default=0.0, metadata={'unit': 'degrees'})
    lat: float = field(default=0.0, metadata={'unit': 'degrees'})


# %%
from dataclasses import fields

fields(Position)


# %%
lat_unit = fields(Position)[2].metadata['unit']
lat_unit


# %%


# %% [markdown]
# How do we make Deck easier to read?
# %% [markdown]
# First let's make PlayingCard easier to read

# %%
@dataclass
class PlayingCard:
    rank: str
    suit: str

    def __str__(self):
        return f'{self.suit}{self.rank}'


# %%
ace_of_spades = PlayingCard('A', '♠')
ace_of_spades


# %%
print(ace_of_spades)


# %%
from dataclasses import dataclass, field
from typing import List

@dataclass
class Deck:
    cards: List[PlayingCard] = field(default_factory=make_french_deck)

    def __repr__(self):
        cards = ', '.join(f'{c!s}' for c in self.cards)
        return f'{self.__class__.__name__}({cards})'


# %%
Deck()


# %%
queen_of_hearts = PlayingCard('Q', '♡')
ace_of_spades = PlayingCard('A', '♠')
ace_of_spades > queen_of_hearts


# %%
@dataclass(order=True)
class PlayingCard:
    rank: str
    suit: str

    def __str__(self):
        return f'{self.suit}{self.rank}'


# %%
queen_of_hearts = PlayingCard('Q', '♡')
ace_of_spades = PlayingCard('A', '♠')
ace_of_spades > queen_of_hearts

# %% [markdown]
# The Ace is larger than the Queen in a deck of cards, so let's fix this.

# %%
RANKS = '2 3 4 5 6 7 8 9 10 J Q K A'.split()
SUITS = '♣ ♢ ♡ ♠'.split()

card = PlayingCard('Q', '♡')
card2 = PlayingCard('J', '♠')

# %% [markdown]
# Since our ranks are sorted in ascending order, we can use their index as a ranking value and we just need to account for ranking the value of suits.

# %%
RANKS.index(card.rank) * len(SUITS) + SUITS.index(card.suit)


# %%
RANKS.index(card2.rank) * len(SUITS) + SUITS.index(card2.suit)


# %%



# %%
from dataclasses import dataclass, field

RANKS = '2 3 4 5 6 7 8 9 10 J Q K A'.split()
SUITS = '♣ ♢ ♡ ♠'.split()

@dataclass(order=True)
class PlayingCard:
    sort_index: int = field(init=False, repr=False)
    rank: str
    suit: str

    def __post_init__(self):
        self.sort_index = (RANKS.index(self.rank) 
                           * len(SUITS) 
                           + SUITS.index(self.suit)
                          )

    def __str__(self):
        return f'{self.suit}{self.rank}'

# %% [markdown]
# Note that .sort_index is added as the **first** field of the class. That way, the comparison is first done using .sort_index and the other fields are used only if there are ties. With field, .sort_index should not be included as a parameter in the .__init__() method (because it is calculated from the .rank and .suit fields). We remove .sort_index from the repr of the class to prevent further confusion.

# %%
queen_of_hearts = PlayingCard('Q', '♡')
ace_of_spades = PlayingCard('A', '♠')
ace_of_spades > queen_of_hearts


# %%
Deck(sorted(make_french_deck()))

# %% [markdown]
# Suppose we need to generate 5 random cards for poker.

# %%
from random import sample

Deck(sample(make_french_deck(), k=5))

# %% [markdown]
# Note that sample works without replacement.
# %% [markdown]
# ## Immutable Dataclasses

# %%
from dataclasses import dataclass

@dataclass(frozen=True)
class Position:
    name: str
    lon: float = 0.0
    lat: float = 0.0


# %%
pos = Position('Oslo', 10.8, 59.9)
pos.name


# %%
pos.name = 'Stockholm'


# %%
from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class ImmutableCard:
    rank: str
    suit: str

@dataclass(frozen=True)
class ImmutableDeck:
    cards: List[ImmutableCard]


# %%
queen_of_hearts = ImmutableCard('Q', '♡')
ace_of_spades = ImmutableCard('A', '♠')
deck = ImmutableDeck([queen_of_hearts, ace_of_spades])
deck


# %%
deck.cards[0] = ImmutableCard('7', '♢')
deck

# %% [markdown]
# Although both dataclasses are immutable, the List is not.  As such, we should use a Tuple instead.
# %% [markdown]
# ## Ordering

# %%
@dataclass
class Position:
    name: str
    lon: float
    lat: float

@dataclass
class Capital(Position):
    country: str


# %%
Capital('Oslo', 10.8, 59.9, 'Norway')


# %%
@dataclass
class Position:
    name: str
    lon: float = 0.0
    lat: float = 0.0

@dataclass
class Capital(Position):
    country: str  # Does NOT work

# %% [markdown]
# Country has no default value and all parameters must have a default value if at least 1 is defined.

# %%
@dataclass
class Position:
    name: str
    lon: float = 0.0
    lat: float = 0.0

@dataclass
class Capital(Position):
    country: str = 'Unknown'
    lat: float = 40.0


# %%
Capital('Madrid', country='Spain')

# %% [markdown]
# Note that the Capital class is superceded by the Position class for ordering.
# %% [markdown]
# ## Slots

# %%
from dataclasses import dataclass

@dataclass
class SimplePosition:
    name: str
    lon: float
    lat: float

@dataclass
class SlotPosition:
    __slots__ = ['name', 'lon', 'lat']
    name: str
    lon: float
    lat: float


# %%
from pympler import asizeof
simple = SimplePosition('London', -0.1, 51.5)
slot = SlotPosition('Madrid', -3.7, 40.4)
asizeof.asizesof(simple, slot)


# %%
from timeit import timeit
timeit('slot.name', setup="slot=SlotPosition('Oslo', 10.8, 59.9)", globals=globals())


# %%
timeit('simple.name', setup="simple=SimplePosition('Oslo', 10.8, 59.9)", globals=globals())

# %% [markdown]
# There is a performance gain from using slots.

