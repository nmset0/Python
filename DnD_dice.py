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
    match dicetype:
        case "d4":
            sides = 4
        case "d6":
            sides = 6
        case "d8":
            sides = 8
        case "d10":
            sides = 10
        case "d12":
            sides = 12
        case "d20":
            sides = 20
        case "stop":
            sides = 0
        case _:
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
    sides = numSides(dicetype)
    dicetype = ''.join(char for char in dicetype if char.isalnum()).lower()    
    match dicetype:
        case "d4":
            # Choose random number between 1 and number of sides of selected die
            roll = random.randint(1, sides) 
            print(f"You rolled a D4 and got a {roll}!")
            message(roll, numSides(dicetype))
        case "d6":
            roll = random.randint(1, sides)
            print(f"You rolled a D6 and got a {roll}!")
            message(roll, numSides(dicetype))
        case "d8":
            roll = random.randint(1, sides)
            print(f"You rolled a D8 and got a {roll}!")
            message(roll, numSides(dicetype))
        case "d10":
            roll = random.randint(1, sides)
            print(f"You rolled a D10 and got a {roll}!")
            message(roll, numSides(dicetype))
        case "d12":
            roll = random.randint(1, sides)
            print(f"You rolled a D12 and got a {random.randint(1, 12)}!")
            message(roll, numSides(dicetype))
        case "d20":
            roll = random.randint(1, sides)
            print(f"You rolled a D20 and got a {roll}!")
            if roll != 20:
                message(roll, numSides(dicetype))
            else:
                print("Amazing!")
        case "stop":
            print(f"\n\"{Goodbye()}\"\n")
        case _:
            print("Invalid die type. Please choose D4, D6, D8, D10, D12, or D20.")




if __name__ == "__main__":
    print("==========[Dungeons & Dragons Dice Roller]==========")
    dnd_roller(diceSelect())