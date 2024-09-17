package weka.api;

//import required classes
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.rules.OneR;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.J48;
import weka.classifiers.Evaluation;
import weka.core.SerializationHelper;

public class Classification{
	public static void main(String args[]) throws Exception{
		
		// load data
		DataSource source = new DataSource("D:\\Year 4\\Data Mining\\Project\\data\\sales_train.arff");
		Instances dataset = source.getDataSet();
		
		// set class index to the last attribute
		dataset.setClassIndex(dataset.numAttributes()-1);
		
		long startTime_oneR = System.currentTimeMillis(); // Record start time
		
		// 1. Apply OneR classifier
        OneR oneR = new OneR();
        oneR.buildClassifier(dataset);
        Evaluation evalOneR = new Evaluation(dataset);
        evalOneR.crossValidateModel(oneR, dataset, 10, new java.util.Random(1)); // 10-fold cross-validation
        
        long endTime_oneR = System.currentTimeMillis(); // Record end time
        long elapsedTime_oneR = endTime_oneR - startTime_oneR; // Calculate elapsed time in milliseconds
        
        // Print out evaluation results for OneR
        System.out.println("=== OneR Evaluation ===");
        System.out.println(evalOneR.toSummaryString());
        System.out.println(evalOneR.toMatrixString());
        System.out.println(evalOneR.toClassDetailsString());
        
        // Print out running time
        System.out.println("Training time: " + elapsedTime_oneR + " milliseconds");
        
        // Save OneR model
//        SerializationHelper.write("D:\\Year 4\\Data Mining\\Project\\models\\prediction\\OneR.model", oneR);
        
		long startTime_NB = System.currentTimeMillis(); // Record start time
        
//		// 2. NaiveBayes Classifier
		NaiveBayes nb = new NaiveBayes();
		nb.buildClassifier(dataset);
        Evaluation eval = new Evaluation(dataset);
        eval.crossValidateModel(nb, dataset, 10, new java.util.Random(1)); // 10-fold cross-validation
		
        long endTime_NB = System.currentTimeMillis(); // Record end time
        long elapsedTime_NB = endTime_NB - startTime_NB; // Calculate elapsed time in milliseconds
        
        // Print out evaluation results
        System.out.println("=== NaiveBayes Evaluation ===");
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toMatrixString());
        System.out.println(eval.toClassDetailsString());
        
        // Print out running time
        System.out.println("Training time: " + elapsedTime_NB + " milliseconds");
        
        // Save NaiveBayes model
//        SerializationHelper.write("D:\\Year 4\\Data Mining\\Project\\models\\prediction\\NaiveBayes.model", nb);
        
		long startTime = System.currentTimeMillis(); // Record start time
		
        // 3. RandomForest classifier
        RandomForest randomForest = new RandomForest();
        randomForest.buildClassifier(dataset);
        Evaluation evalRandomForest = new Evaluation(dataset);
        evalRandomForest.crossValidateModel(randomForest, dataset, 10, new java.util.Random(1)); // 10-fold cross-validation

        long endTime = System.currentTimeMillis(); // Record end time
        long elapsedTime = endTime - startTime; // Calculate elapsed time in milliseconds
        
        // Print out evaluation results for RandomForest
        System.out.println("=== RandomForest Evaluation ===");
        System.out.println(evalRandomForest.toSummaryString());
        System.out.println(evalRandomForest.toMatrixString());
        System.out.println(evalRandomForest.toClassDetailsString());
        
        // Print out running time
        System.out.println("Training time: " + elapsedTime + " milliseconds");
//		
        // Save RandomForest model
//        SerializationHelper.write("D:\\Year 4\\Data Mining\\Project\\models\\prediction\\RandomForest.model", randomForest);
	}
}