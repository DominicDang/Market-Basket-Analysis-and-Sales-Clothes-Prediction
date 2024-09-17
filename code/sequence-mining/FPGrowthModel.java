package sequence.mining;

import weka.associations.AssociationRule;
import weka.core.SerializationHelper;
import weka.associations.FPGrowth;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.util.List;

public class FPGrowthModel {
	public static void main(String args[]) throws Exception{
		//load data
		String dataset = "D:\\Year 4\\Data Mining\\Project\\data\\filtered_basket_sets.arff";
		DataSource source = new DataSource(dataset);
		Instances data = source.getDataSet();
		
		long startTime = System.currentTimeMillis(); // Record start time
		
		//the FPGrowth algorithm
		FPGrowth model = new FPGrowth();
        String[] options = {"-P", "2", "-I", "-1", "-N", "10", "-T", "0", "-C", "0.2", "-M", "0.01"};
        model.setOptions(options);
        
        // Build associations
        model.buildAssociations(data);
        
		long endTime = System.currentTimeMillis(); // Record end time
        long elapsedTime = endTime - startTime; // Calculate elapsed time in milliseconds  
		System.out.println(model);
        System.out.println("Training time: " + elapsedTime + " milliseconds");
		
		
		// Get association rules
        List<AssociationRule> rules = model.getAssociationRules().getRules();
        
        // Print performance metrics for each association rule
        System.out.println("Association Rules:");
        for (AssociationRule rule : rules) {
            System.out.println("Rule: " + rule.getPremise() + " => " + rule.getConsequence());
            System.out.println("Support: " + rule.getTotalSupport());
            System.out.println();
        }
        
        // Save FPGrowth model
//      SerializationHelper.write("D:\\Year 4\\Data Mining\\Project\\models\\sequence mining\\FPGrowth.model", model);
	}
}
