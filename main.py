import FunctionOne.FunctionOneThreeLayerNeuralNetwork as function_one_single_hidden
import FunctionOne.FunctionOneFourLayerNeuralNetwork as function_one_multi_hidden
import FunctionTwo.FunctionTwoThreeLayerNeuralNetwork as function_two_single_hidden
import FunctionTwo.FunctionTwoFourLayerNeuralNetwork as function_two_multi_hidden

# csv_writer = open("BestAndWorst.csv", "w")
# csv_writer.write("\nneurons, epochs, mse, execution_time")
# csv_writer.write("\nFunction_One_Single_Hidden_Worst")
# function_one_single_hidden_worst = function_one_single_hidden.NeuralNetwork(100, 1, csv_writer)
# function_one_single_hidden_worst.plot_error()
# function_one_single_hidden_worst.test_network()
# csv_writer.write("\nFunction_One_Single_Hidden_Best")
# function_one_single_hidden_best = function_one_single_hidden.NeuralNetwork(1000, 10, csv_writer)
# function_one_single_hidden_best.plot_error()
# function_one_single_hidden_best.test_network()
# csv_writer.write("\nFunction_One_Multi_Hidden_Worst")
# function_one_multi_hidden_worst = function_one_multi_hidden.NeuralNetwork(100, 1, csv_writer)
# function_one_multi_hidden_worst.plot_error()
# function_one_multi_hidden_worst.test_network()
# csv_writer.write("\nFunction_One_Multi_Hidden_Best")
# function_one_multi_hidden_best = function_one_multi_hidden.NeuralNetwork(1000, 10, csv_writer)
# function_one_multi_hidden_best.plot_error()
# function_one_multi_hidden_best.test_network()
# csv_writer.write("\nFunction_Two_Single_Hidden_Worst")
# function_two_single_hidden_worst = function_two_single_hidden.NeuralNetwork(100, 1, csv_writer)
# function_two_single_hidden_worst.plot_error()
# function_two_single_hidden_worst.test_network()
# csv_writer.write("\nFunction_Two_Single_Hidden_Best")
# function_two_single_hidden_best = function_two_single_hidden.NeuralNetwork(1000, 10, csv_writer)
# function_two_single_hidden_best.plot_error()
# function_two_single_hidden_best.test_network()
# csv_writer.write("\nFunction_Two_Multi_Hidden_Worst")
# function_two_multi_hidden_worst = function_two_multi_hidden.NeuralNetwork(100, 1, csv_writer)
# function_two_multi_hidden_worst.plot_error()
# function_two_multi_hidden_worst.test_network()
# csv_writer.write("\nFunction_Two_Multi_Hidden_Best")
# function_two_multi_hidden_best = function_two_multi_hidden.NeuralNetwork(1000, 10, csv_writer)
# function_two_multi_hidden_best.plot_error()
# function_two_multi_hidden_best.test_network()

csv_writer = open("ProjectStatistics.csv", "w")
csv_writer.write("Function_One_Single_Hidden")
csv_writer.write("\nneurons, epochs, mse, execution_time")
function_one_single_hidden.NeuralNetwork(100, 1, csv_writer)
function_one_single_hidden.NeuralNetwork(200, 2, csv_writer)
function_one_single_hidden.NeuralNetwork(300, 3, csv_writer)
function_one_single_hidden.NeuralNetwork(400, 4, csv_writer)
function_one_single_hidden.NeuralNetwork(500, 5, csv_writer)
function_one_single_hidden.NeuralNetwork(600, 6, csv_writer)
function_one_single_hidden.NeuralNetwork(700, 7, csv_writer)
function_one_single_hidden.NeuralNetwork(800, 8, csv_writer)
function_one_single_hidden.NeuralNetwork(900, 9, csv_writer)
function_one_single_hidden.NeuralNetwork(1000, 10, csv_writer)
csv_writer.write("\nFunction_One_Multi_Hidden")
csv_writer.write("\nneurons, epochs, mse, execution_time")
function_one_multi_hidden.NeuralNetwork(100, 1, csv_writer)
function_one_multi_hidden.NeuralNetwork(200, 2, csv_writer)
function_one_multi_hidden.NeuralNetwork(300, 3, csv_writer)
function_one_multi_hidden.NeuralNetwork(400, 4, csv_writer)
function_one_multi_hidden.NeuralNetwork(500, 5, csv_writer)
function_one_multi_hidden.NeuralNetwork(600, 6, csv_writer)
function_one_multi_hidden.NeuralNetwork(700, 7, csv_writer)
function_one_multi_hidden.NeuralNetwork(800, 8, csv_writer)
function_one_multi_hidden.NeuralNetwork(900, 9, csv_writer)
function_one_multi_hidden.NeuralNetwork(1000, 10, csv_writer)
csv_writer.write("\nFunction_Two_Single_Hidden")
csv_writer.write("\nneurons, epochs, mse, execution_time")
function_two_single_hidden.NeuralNetwork(100, 1, csv_writer)
function_two_single_hidden.NeuralNetwork(200, 2, csv_writer)
function_two_single_hidden.NeuralNetwork(300, 3, csv_writer)
function_two_single_hidden.NeuralNetwork(400, 4, csv_writer)
function_two_single_hidden.NeuralNetwork(500, 5, csv_writer)
function_two_single_hidden.NeuralNetwork(600, 6, csv_writer)
function_two_single_hidden.NeuralNetwork(700, 7, csv_writer)
function_two_single_hidden.NeuralNetwork(800, 8, csv_writer)
function_two_single_hidden.NeuralNetwork(900, 9, csv_writer)
function_two_single_hidden.NeuralNetwork(1000, 10, csv_writer)
csv_writer.write("\nFunction_Two_Multi_Hidden")
csv_writer.write("\nneurons, epochs, mse, execution_time")
function_two_multi_hidden.NeuralNetwork(100, 1, csv_writer)
function_two_multi_hidden.NeuralNetwork(200, 2, csv_writer)
function_two_multi_hidden.NeuralNetwork(300, 3, csv_writer)
function_two_multi_hidden.NeuralNetwork(400, 4, csv_writer)
function_two_multi_hidden.NeuralNetwork(500, 5, csv_writer)
function_two_multi_hidden.NeuralNetwork(600, 6, csv_writer)
function_two_multi_hidden.NeuralNetwork(700, 7, csv_writer)
function_two_multi_hidden.NeuralNetwork(800, 8, csv_writer)
function_two_multi_hidden.NeuralNetwork(900, 9, csv_writer)
function_two_multi_hidden.NeuralNetwork(1000, 10, csv_writer)