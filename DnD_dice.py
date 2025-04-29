import random

# Prompt the user what die they want to roll
def diceSelect():
    dicetype = input("Which die would you like to roll? Type \"STOP\" to exit.\n")
    # Expects input D4, D6, D8, D10, D12, D20
    return dicetype

# Get the number of sides of the chosen die
def numSides(dicetype):
    # ignore case and remove any punctuation
    dicetype = ''.join(char for char in dicetype if char.isalnum()).lower()
    sides = dice_map.get(dicetype)
    if sides is None:
        print("Invalid die type. Please choose d4, d6, d8, d10, d12, or d20.")
    return sides
    
# Fun things
def message(roll, sides):
    if roll >= sides/2:
        print("Onward, adventurer!")
    else:
        print("It's over. o_o")

def Goodbye():
    stringList = [
    "Every breath is a gift, and every moment we spend is a reminder of our mortality.",
    "Life is a fragile flower, its beauty born from the knowledge that it will wither.",
    "Our time here is a fleeting dream, a whisper in the vast expanse of eternity.",
    "The only constant in life is change, and the only certainty is death.",
    "We are all just dust, and one day we will return to dust."
    ]
    randString = random.choice(stringList) # Get random string from list
    return(randString)
    

# Roll the die
def dnd_roller(dicetype):
    # Ignore case and remove any punctuation
    dicetype = ''.join(char for char in dicetype if char.isalnum()).lower()

    sides = dice_map.get(dicetype)
    if sides is None:
        print("Invalid die type. Please choose D4, D6, D8, D10, D12, or D20.")
        return
    if dicetype == "stop":
        print(f"\n\"{Goodbye()}\"\n")
        return
    
    roll = random.randint(1, sides)
    print(f"You rolled a {dicetype.upper()} and got a {roll}!")
    if dicetype == "d20" and roll == 20:
        print("Amazing!")
    else:
        message(roll, sides)




if __name__ == "__main__":
    print("==========[Dungeons & Dragons Dice Roller]==========")
    dice_map = {
        "d4": 4,
        "d6": 6,
        "d8": 8,
        "d10": 10,
        "d12": 12,
        "d20": 20,
        "stop": 0
    }    
    dnd_roller(diceSelect())