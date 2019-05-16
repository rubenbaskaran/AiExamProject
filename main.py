import FunctionOne.FunctionOneThreeLayerNeuralNetwork as function_one_single_hidden
import FunctionOne.FunctionOneFourLayerNeuralNetwork as function_one_multi_hidden
import FunctionTwo.FunctionTwoThreeLayerNeuralNetwork as function_two_single_hidden


csv_writer = open("ProjectStatistics.csv", "w")
csv_writer.write("Function_One_Single_Hidden")
csv_writer.write("\nneurons, epochs, mse, execution_time")
function_one_single_hidden.NeuralNetwork(100, 10, csv_writer)
function_one_single_hidden.NeuralNetwork(100, 20, csv_writer)
function_one_single_hidden.NeuralNetwork(100, 30, csv_writer)
csv_writer.write("\nFunction_One_Multi_Hidden")
csv_writer.write("\nneurons, epochs, mse, execution_time")
function_one_multi_hidden.NeuralNetwork(100, 10, csv_writer)
function_one_multi_hidden.NeuralNetwork(100, 20, csv_writer)
function_one_multi_hidden.NeuralNetwork(100, 30, csv_writer)
csv_writer.write("\nFunction_Two_Single_Hidden")
csv_writer.write("\nneurons, epochs, mse, execution_time")
function_two_single_hidden.NeuralNetwork(100, 10, csv_writer)
function_two_single_hidden.NeuralNetwork(100, 20, csv_writer)
function_two_single_hidden.NeuralNetwork(100, 30, csv_writer)


