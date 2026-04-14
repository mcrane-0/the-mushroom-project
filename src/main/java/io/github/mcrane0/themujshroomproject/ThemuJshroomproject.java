/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 */

package io.github.mcrane0.themujshroomproject;

import java.io.IOException;
import java.nio.file.FileSystems;
import java.nio.file.Path;
import java.util.Properties;
import smile.classification.Classifier;
import smile.io.*;
import smile.data.*;
import smile.classification.KNN;
import smile.classification.LinearSVM;
import smile.classification.SVM;
import smile.neighbor.KDTree;

/**
 *
 * @author miles
 */
public class ThemuJshroomproject {
    
    public static void main(String[] args) throws IOException {   
        // get data from CSV
        CSV carDataCSV = new CSV();
        Path pathToCSV = FileSystems.getDefault().getPath("car.data");
        DataFrame carDFRaw = carDataCSV.read(pathToCSV);
        System.out.println("carDFRaw:\n" + carDFRaw.toString());
        
        int[] carDFClasses = classesToNumbers(carDFRaw.column("V7").toStringArray());
        DataFrame carDF = carDFRaw.drop("V7");
        
        valuesToNumbers(carDF);
        System.out.println("carDF:\n" + carDF.toString());
        
        
        // KNN implementation and testing
        int neighbors = 1;
        KNN knnThing = new KNN(KDTree.of(carDF.toArray()), carDFClasses, neighbors);
        System.out.println("\nKNN, NEIGHBORS: " + neighbors);
        testKNN(knnThing, carDF, carDFClasses);
        
        neighbors = 3;
        knnThing = new KNN(KDTree.of(carDF.toArray()), carDFClasses, neighbors);
        System.out.println("\nKNN, NEIGHBORS: " + neighbors);
        testKNN(knnThing, carDF, carDFClasses);
        
        neighbors = 5;
        knnThing = new KNN(KDTree.of(carDF.toArray()), carDFClasses, neighbors);
        System.out.println("\nKNN, NEIGHBORS: " + neighbors);
        testKNN(knnThing, carDF, carDFClasses);
        
        neighbors = 7;
        knnThing = new KNN(KDTree.of(carDF.toArray()), carDFClasses, neighbors);
        System.out.println("\nKNN, NEIGHBORS: " + neighbors);
        testKNN(knnThing, carDF, carDFClasses);
        
        neighbors = 10;
        knnThing = new KNN(KDTree.of(carDF.toArray()), carDFClasses, neighbors);
        System.out.println("\nKNN, NEIGHBORS: " + neighbors);
        testKNN(knnThing, carDF, carDFClasses);
        
        
        // SVM implementation and testing
        svmOneAgainstMany((carDF.toArray()), carDFClasses);
    }
    
    public static int[] classesToNumbers(String[] classesRaw){
        int[] numClass = new int[classesRaw.length];
        for (int i = 0; i < classesRaw.length; i++){
            // V7 - class:  unacc, acc, good, vgood
            String value = classesRaw[i];
            switch (value) {
                case "unacc" -> numClass[i] = 0;
                case "acc" -> numClass[i] = 1;
                case "good" -> numClass[i] = 2;
                default -> // value = "vgood"
                    numClass[i] = 3;
            }
        }
        return numClass;
    }
    
    public static void valuesToNumbers(DataFrame df){
        for (int i = 0; i < df.size(); i++){ 
            // V1 - buying:   vhigh, high, med, low.
            String value = df.get(i, 0).toString();
            switch (value) {
                case "vhigh" -> df.set(i, 0, "0");
                case "high" -> df.set(i, 0, "1");
                case "med" -> df.set(i, 0, "2");
                default -> // value = "low"
                    df.set(i, 0, "3");
            }
            
            // V2 - maint:    vhigh, high, med, low.
            value = df.get(i, 1).toString();
            switch (value) {
                case "vhigh" -> df.set(i, 1, "0");
                case "high" -> df.set(i, 1, "1");
                case "med" -> df.set(i, 1, "2");
                default -> // value = "low"
                    df.set(i, 1, "3");
            }
            
            // V3 - doors:    2, 3, 4, 5more.
            value = df.get(i, 2).toString();
            switch (value) {
                case "2" -> df.set(i, 2, "0");
                case "3" -> df.set(i, 2, "1");
                case "4" -> df.set(i, 2, "2");
                default -> // value = "5more"
                    df.set(i, 2, "3");
            }
            
            // V4 - persons:  2, 4, more.
            value = df.get(i, 3).toString();
            switch (value) {
                case "2" -> df.set(i, 3, "0");
                case "4" -> df.set(i, 3, "1");
                default -> // value = "more"
                    df.set(i, 3, "2");
            }
            
            // V5 - lug_boot: small, med, big.
            value = df.get(i, 4).toString();
            switch (value) {
                case "small" -> df.set(i, 4, "0");
                case "med" -> df.set(i, 4, "1");
                default -> // value = "big"
                    df.set(i, 4, "2");
            }
            
            // V6 - safety:   low, med, high.
            value = df.get(i, 5).toString();
            switch (value) {
                case "low" -> df.set(i, 5, "0");
                case "med" -> df.set(i, 5, "1");
                default -> // value = "high"
                    df.set(i, 5, "2");
            }
        }
    } 
    
    public static void testKNN(KNN model, DataFrame df, int[] classes){
        int predCorrect = 0;
        int predIncorrect = 0;
        for (int i = 0; i < df.size(); i++){
            int prediction = model.predict(df.get(i).toArray());
            //System.out.print("DEBUG:\ti=" + i + "\tpred=" + prediction + "\ttrue=" + classes[i]);
            if (prediction == classes[i]){
                predCorrect++;
                //System.out.println("\tcorrect.");
            }   
            else{
                predIncorrect++;
                //System.out.println("\tincorrect.");
            }
        }
        System.out.println("RESULTS:"
                            + "\nCorrect Predictions:\t" + predCorrect
                            + "\nIncorrect:\t\t" + predIncorrect);
    }
    
    public static void svmOneAgainstMany(double[][] data, int[] classes){
        
    }

}
