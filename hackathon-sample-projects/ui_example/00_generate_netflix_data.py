import pandas as pd
import os

# Create a list of popular Netflix shows with their descriptions
netflix_shows = [
    {
        "title": "Stranger Things",
        "genre": "Sci-Fi, Horror",
        "year": 2016,
        "seasons": 4,
        "description": "In a small town where everyone knows everyone, a peculiar incident starts a chain of events that leads to the disappearance of a child, which begins to tear at the fabric of an otherwise peaceful community. Dark supernatural forces are at play and something otherworldly has entered the scene. A group of kids, along with an emotionally troubled police chief and the missing boy's mom, must confront terrifying forces in order to save their town."
    },
    {
        "title": "The Crown",
        "genre": "Drama, History",
        "year": 2016,
        "seasons": 6,
        "description": "This drama follows the political rivalries and romance of Queen Elizabeth II's reign and the events that shaped the second half of the 20th century. As the decades pass, personal intrigues, scandals, and political rivalries are revealed that shaped the destiny of not only the royal family but the world beyond the palace walls."
    },
    {
        "title": "Squid Game",
        "genre": "Drama, Thriller",
        "year": 2021,
        "seasons": 1,
        "description": "Hundreds of cash-strapped players accept a strange invitation to compete in children's games. Inside, a tempting prize awaits with deadly high stakes. A survival game that has a whopping 45.6 billion-won prize at stake. The desperate participants engage in deadly childhood games to win the prize, only to realize the deadly consequences of losing."
    },
    {
        "title": "Money Heist",
        "genre": "Crime, Drama",
        "year": 2017,
        "seasons": 5,
        "description": "A criminal mastermind who goes by 'The Professor' has a plan to pull off the biggest heist in recorded history -- to print billions of euros in the Royal Mint of Spain. To help him carry out the ambitious plan, he recruits eight people with certain abilities and who have nothing to lose. The group of thieves take hostages to aid in their negotiations with the authorities, who strategize to come up with a way to capture The Professor."
    },
    {
        "title": "Bridgerton",
        "genre": "Drama, Romance",
        "year": 2020,
        "seasons": 2,
        "description": "The eight close-knit siblings of the Bridgerton family look for love and happiness in London high society. Set in the lavish and competitive world of Regency London during the season when debutantes are presented at court, this period drama centered on the Bridgerton family offers a glimpse into the sophisticated, sexy, and sometimes scandalous lives of British high society."
    },
    {
        "title": "The Witcher",
        "genre": "Fantasy, Action",
        "year": 2019,
        "seasons": 3,
        "description": "Geralt of Rivia, a solitary monster hunter, struggles to find his place in a world where people often prove more wicked than beasts. Mutated by a mysterious ritual called the Trial of the Grasses, Geralt possesses superhuman reflexes and strength, and is a master swordsman. He wanders the Continent looking for work, hunting down dangerous creatures that threaten people's lives, always for a price."
    },
    {
        "title": "Ozark",
        "genre": "Crime, Drama",
        "year": 2017,
        "seasons": 4,
        "description": "A financial adviser drags his family from Chicago to the Missouri Ozarks, where he must launder money for a Mexican drug cartel to keep his family safe. He tangles with local criminals and eventually must contend with the Kansas City Mafia. The dark undertones and dire circumstances create tension and a constant state of suspense as the Byrde family navigates their dangerous new home."
    },
    {
        "title": "Black Mirror",
        "genre": "Sci-Fi, Thriller",
        "year": 2011,
        "seasons": 6,
        "description": "An anthology series exploring a twisted, high-tech multiverse where humanity's greatest innovations and darkest instincts collide. Each standalone episode examines modern society, particularly with regard to the unanticipated consequences of new technologies. The show often has dark and satirical tones, and features elements of science fiction, psychological horror, and dystopian fiction."
    },
    {
        "title": "Wednesday",
        "genre": "Comedy, Fantasy",
        "year": 2022,
        "seasons": 1,
        "description": "Wednesday Addams is sent to Nevermore Academy, a peculiar boarding school where she attempts to master her emerging psychic ability, thwart a killing spree terror that has plagued the local town, and solve the supernatural mystery that embroiled her parents 25 years ago—all while navigating her new relationships at Nevermore."
    },
    {
        "title": "Dark",
        "genre": "Sci-Fi, Mystery",
        "year": 2017,
        "seasons": 3,
        "description": "A family saga with a supernatural twist, set in a German town where the disappearance of two young children exposes the relationships among four families. The story takes place in several different time periods, all intrinsically interconnected through elaborate time travel plot lines. As the mystery unravels, it becomes clear that a complex conspiracy spans multiple generations."
    }
]

# Create DataFrame
netflix_df = pd.DataFrame(netflix_shows)

# Make sure the data directory exists
if not os.path.exists("data"):
    os.makedirs("data")
    print("Created directory: data")

# Save to CSV in the data directory
netflix_df.to_csv("data/netflix_reviews.csv", index=False)
print("Saved Netflix show descriptions to data/netflix_reviews.csv")

