# Object detection generator and counter

requirement were specified in requirement.txt
using virtualenv and python3.8.6


Explaination:

Generator.py:
    Generate first task output.
    $python Generator.py             
    This will print the demanded output

    $python Generator.py display
    This will render the generate result, press "N" for next and "ESC" to leave

    Generator.getGroundTruth() will return a dict of {imgID:[catagory] } as groundTruth
    
    Generator.getDetectionRes() will return a list of demanded output, use param getDetectionRes(useObj=True) to directly pass a list of detection object.

mainApp.py:
    main entry point of second task, connecting generated output to tracking and counting system.

    $python mainApp.py  will output the catagory detected by frame

    getCountingResult() will return the same format to the generator groundTruth for comparison

tester.py::
    some test function to format result and plot the distribution.
    
    $python tester.py will print the count of 5 catagory and the groundTruth.
    simultaneously showing the plot for frame and catagory detected.
    
     



