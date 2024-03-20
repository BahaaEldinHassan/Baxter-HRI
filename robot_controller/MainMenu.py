import initaliseArms
import subprocess, os
import time as t

## Class that contains the functions to run each of the options within the menu ##
class MainMenu:

    ## Opens the head camera in a seperate termial ##
    def openHead(self):
        ## Subprocess ["gnome-terminal", "--"] allows the user to open the camera while still being able to use the main menu ##
        subprocess.call(["gnome-terminal", "--", "python3", "headCamera.py"])

    ## Opens the left arm camera in a seperate termial ##
    def openLeft(self):
        ## Subprocess ["gnome-terminal", "--"] allows the user to open the camera while still being able to use the main menu ##
        subprocess.call(["gnome-terminal", "--", "python3", "LeftArmCamera.py"])

    ## Opens the right arm camera in a seperate termial ##
    def openRight(self):
        ## Subprocess ["gnome-terminal", "--"] allows the user to open the camera while still being able to use the main menu ##
        subprocess.call(["gnome-terminal", "--", "python3", "RightArmCamera.py"])

    ## Starts the manual control function to be able to control the arm with the space mouse ##
    def maualControl(self):
        subprocess.call(["python3", "maualControl.py"])

    ## Start the autonimous control function ##
    def autonimousControl(self):
        self.performAction()

    ## Starts the response actions function
    def performAction(self):
        ## Gets the file path ##
        filepath = os.path.dirname(os.path.abspath(__file__))
        ## Reads the file that contains the recognised actions from the computer vision ##
        file = open(filepath+"/action.txt", "r")
        action = file.read()
        file.close()
        ## Sends the value to the action node for it to determin which response action should be performed ##
        subprocess.call(["python3", "actions.py", "--action", action[0]])

    ## Initalises the arms back to a default position ##
    def initaliseArms(self):
        initaliseArms.main()
    
    ## Closes the progam & prints message ##
    def exit(self):
        exit("Close")

    ## The main menu function which will be displayed to the user ##
    def mainMenu(self):

        ## Makes the class iterable ##
        MM = MainMenu()

        ## The users input their choice from the list below
        option = input("################################################################################\n"
                        "    Baxter Robot Control \n"
                        "################################################################################"
                        "\n  - 1: Initalise Arms:"
                        "\n  - 2: Open Head Camera:"
                        "\n  - 3: Open Left Arm Camera:"
                        "\n  - 4: Open Right Arm Camera:"
                        "\n  - 5: Manual Control:"
                        "\n  - 6: Autonimous Control:"
                        "\n  - Q: Exit \n"
                        "################################################################################\n"
                        "\n"
                        "################################################################################\n"
                        "\n  - Option: ")

        ## Infinite Loop ##
        while True:
            ## Try a valid response first ##
            try:
                ## If the user wants to initalise the arms then they enter 1 ##
                if option == "1":
                    print("Initalizing Arms")
                    t.sleep(1.0)
                    ## Runs the intailise arms function ##
                    MM.initaliseArms()
                    ## Time sleep allows enough time for the subprocess to complete before enabling the main menu again ##
                    t.sleep(5.0)
                    ## Iterates back to the begining of the class so the main menu is presented to the user again for them to select another option ##
                    MM.mainMenu()
                    break
                ## If the user wants to open the head camera then they enter 2 ##
                elif option == "2":
                    print("Open Head Hand Camera")
                    t.sleep(1.0)
                    ## Runs the intailise arms function ##
                    MM.openHead()
                    ## Time sleep allows enough time for the subprocess to complete before enabling the main menu again ##
                    t.sleep(5.0)
                    ## Iterates back to the begining of the class so the main menu is presented to the user again for them to select another option ##
                    MM.mainMenu()
                    break
                ## If the user wants to open left hand camera then they enter 3 ##
                elif option == "3":
                    print("Open Left Hand Camera")
                    t.sleep(1.0)
                    MM.openLeft()
                    ## Time sleep allows enough time for the subprocess to complete before enabling the main menu again ##
                    t.sleep(5.0)
                    ## Iterates back to the begining of the class so the main menu is presented to the user again for them to select another option ##
                    MM.mainMenu()
                    break
                ## If the user wants to open right hand camera then they enter 4 ##
                elif option == "4":
                    print("Open Right Hand Camera")
                    t.sleep(1.0)
                    MM.openRight()
                    ## Time sleep allows enough time for the subprocess to complete before enabling the main menu again ##
                    t.sleep(5.0)
                    ## Iterates back to the begining of the class so the main menu is presented to the user again for them to select another option ##
                    MM.mainMenu()
                    break
                ## If the user wants to use the maual control mode then they enter 5 ##
                elif option == "5":
                    print("Maual Control")
                    t.sleep(0.5)
                    MM.maualControl()
                    ## Time sleep allows enough time for the subprocess to complete before enabling the main menu again ##
                    t.sleep(5.0)
                    ## Iterates back to the begining of the class so the main menu is presented to the user again for them to select another option ##
                    MM.mainMenu()
                    break
                ## If the user wants to use the autonimous mode then they enter 6 ##
                elif option == "6":
                    print("Automimous Control")
                    t.sleep(0.5)
                    MM.autonimousControl()
                    ## Time sleep allows enough time for the subprocess to complete before enabling the main menu again ##
                    t.sleep(5.0)
                    ## Iterates back to the begining of the class so the main menu is presented to the user again for them to select another option ##
                    MM.mainMenu()
                    break
                ## If the user wants to quit the program then they enter q ##
                elif option == "q":
                    print("Quitting the program")
                    print("Goodbye!")
                    MM.exit()
                    break
                ## If a user does not enter a valid option the main menu will print out a error message & loop back to allow the user try again ##
                else:
                    print("Please enter a valid option")
                    option = MM.mainMenu()
            ## If user uses Ctl+C it will close the program ##
            except KeyboardInterrupt:
                    exit("close")

## --MAIN PROGRAM-- ##
def main():
    mm = MainMenu()
    mm.mainMenu()

if __name__ == "__main__":
    main()
