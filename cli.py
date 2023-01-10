import sys
import click
import warnings

from tkinter import filedialog as fd


from logic.train import train_model
from logic.segment import segment_image


warnings.filterwarnings('ignore') # setting ignore as a parameter


@click.group()
def cli():
    pass

@click.command()
def train_model_unet():
    acknowledgement = input("Write Yes if you are sure to do it: ")

    if acknowledgement == 'Yes':
        try:
            images = fd.askdirectory()
            labels = fd.askdirectory()
            
            if not images or not labels:
                raise Exception

            if train_model(images, labels):
                click.secho("The training was finished successfully.", fg='green')
        except Exception:  
            click.secho("Please select the images path and the labels path or check the image type and try again", fg='yellow')

@click.command()
def segment_unet():
    try:
        f = fd.askopenfilename()
        o = fd.askdirectory()

        if not f or not o :
            raise Exception

        if segment_image(f,o):
            click.secho("The image: " +f+ " was segmented successfully.", fg='green')
    except Exception:  
        click.secho("Please select an image and a output path or check the image type and try again", fg='yellow')

@click.command()
def exit():
    sys.exit()


if __name__ == '__main__':
    cli.add_command(exit)
    cli.add_command(train_model_unet)
    cli.add_command(segment_unet)
    click.secho("Welcome to model training and image segmentation prototype!", fg='yellow')
    
    while True:
        
        command_list = cli.list_commands(click.Context)
        print("")
        try:
            for i in range(len(command_list)):
                print(str(i+1) + "-" + command_list[i])
            command_option = int(input("Please enter a command number: "))
            if command_option <= len(command_list) and command_option > 0:
                cli.main([command_list[command_option-1]],
                    standalone_mode=False)
            else:
                raise ValueError
        except ValueError:
            click.secho("That option doesn't exist, please try again!", fg='yellow')