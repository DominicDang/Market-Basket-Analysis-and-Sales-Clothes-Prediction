package weka.api;

import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.core.converters.ArffSaver;
import java.io.File;
import weka.core.converters.ConverterUtils.DataSource;

public class AttrSelection{
	public static void main(String args[]) throws Exception{
		//load data
		DataSource source = new DataSource("D:\\Year 4\\Data Mining\\Project\\data\\filtered_sales_clothes.arff");
		Instances filteredData = source.getDataSet();
		
        // Set the index of the target attribute
        int targetAttributeIndex = 2;
        filteredData.setClassIndex(targetAttributeIndex);

		AttributeSelection filter = new AttributeSelection();
		
		//create evaluator and search algorithm objects
		CfsSubsetEval eval = new CfsSubsetEval();
		GreedyStepwise search = new GreedyStepwise();
		search.setSearchBackwards(true);
		filter.setEvaluator(eval);
		filter.setSearch(search);
		filter.setInputFormat(filteredData);
		
		Instances newData = Filter.useFilter(filteredData, filter);
		
		ArffSaver saver = new ArffSaver();
		saver.setInstances(newData);
		saver.setFile(new File("D:\\Year 4\\Data Mining\\Project\\data\\dimension_reduction_sales_clothes.arff"));
		saver.writeBatch();
	}
}


