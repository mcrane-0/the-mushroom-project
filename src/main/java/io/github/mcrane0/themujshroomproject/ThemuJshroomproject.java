/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 */

package io.github.mcrane0.themujshroomproject;

import java.io.IOException;
import java.nio.file.FileSystems;
import java.nio.file.Path;
import java.util.Arrays;
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
        svmOneAgainstMany((carDF.toArray()), carDFClasses, 7.5);
        svmOneAgainstMany((carDF.toArray()), carDFClasses, 10.0);
        svmOneAgainstMany((carDF.toArray()), carDFClasses, 12.5);
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
        for (int i = 0; i < df.size(); i++){
            int prediction = model.predict(df.get(i).toArray());
            //System.out.print("DEBUG:\ti=" + i + "\tpred=" + prediction + "\ttrue=" + classes[i]);
            if (prediction == classes[i])
                predCorrect++;
        }
        System.out.println("RESULTS:"
                            + "\nCorrect Predictions:\t" + predCorrect
                            + "\nIncorrect:\t\t" + (df.size() - predCorrect));
    }
    
    public static void svmOneAgainstMany(double[][] data, int[] classes, double softMargin){
        // V7 - class:  unacc, acc, good, vgood
        int[] svmPreds = new int[classes.length]; // final predictions
        Arrays.fill(svmPreds, -1);
        
        // 0 against all
        int[] zeroAgainstAllClasses = new int[classes.length];
        int[] zeroAgainstAllPreds = new int[classes.length]; // predictions
        double[] zeroAgainstAllPredsScore = new double[classes.length]; // score/confidence of predictions

        for (int i = 0; i < classes.length; i++){
            if (classes[i] == 0) // class = unacc
                zeroAgainstAllClasses[i] = 1;
            else // class = (anything else)
                zeroAgainstAllClasses[i] = -1;
        }
        LinearSVM zeroAgainstAll = SVM.fit(data, zeroAgainstAllClasses, new SVM.Options(softMargin));
        int predCorrect = 0;
        for (int j = 0; j < data.length; j++){
            zeroAgainstAllPreds[j] = zeroAgainstAll.predict(data[j]);
            zeroAgainstAllPredsScore[j] = zeroAgainstAll.score(data[j]);
            if (zeroAgainstAllPreds[j] == zeroAgainstAllClasses[j])
                predCorrect++;
        }
        System.out.println("\nSVM: 0 AGAINST ALL:"
                            + "\nCorrect:\t" + predCorrect
                            + "\nIncorrect:\t" + (data.length - predCorrect));
        for (int k = 0; k < data.length; k++){
            if (zeroAgainstAllPreds[k] == 1)
                svmPreds[k] = 0;
        }

        // 1 against all
        int[] oneAgainstAllClasses = new int[classes.length];
        int[] oneAgainstAllPreds = new int[classes.length]; // predictions
        double[] oneAgainstAllPredsScore = new double[classes.length]; // score/confidence of predictions

        for (int i = 0; i < classes.length; i++){
            if (classes[i] == 1) // class = acc
                oneAgainstAllClasses[i] = 1;
            else // class = (anything else)
                oneAgainstAllClasses[i] = -1;
        }
        LinearSVM oneAgainstAll = SVM.fit(data, oneAgainstAllClasses, new SVM.Options(softMargin));
        predCorrect = 0;
        for (int j = 0; j < data.length; j++){
            oneAgainstAllPreds[j] = oneAgainstAll.predict(data[j]);
            oneAgainstAllPredsScore[j] = oneAgainstAll.score(data[j]);
            if (oneAgainstAllPreds[j] == oneAgainstAllClasses[j])
                predCorrect++;
        }
        System.out.println("\nSVM: 1 AGAINST ALL:"
                            + "\nCorrect:\t" + predCorrect
                            + "\nIncorrect:\t" + (data.length - predCorrect));
        for (int k = 0; k < data.length; k++){
            if (oneAgainstAllPreds[k] == 1){
                if (svmPreds[k] == 0 && oneAgainstAllPredsScore[k] <= zeroAgainstAllPredsScore[k]){
                    svmPreds[k] = 0;
                }
                else {
                    svmPreds[k] = 1;
                }
            }
        }
        
        // 2 against all
        int[] twoAgainstAllClasses = new int[classes.length];
        int[] twoAgainstAllPreds = new int[classes.length]; // predictions
        double[] twoAgainstAllPredsScore = new double[classes.length]; // score/confidence of predictions

        for (int i = 0; i < classes.length; i++){
            if (classes[i] == 2) // class = good
                twoAgainstAllClasses[i] = 1;
            else // class = (anything else)
                twoAgainstAllClasses[i] = -1;
        }
        LinearSVM twoAgainstAll = SVM.fit(data, twoAgainstAllClasses, new SVM.Options(softMargin));
        predCorrect = 0;
        for (int j = 0; j < data.length; j++){
            twoAgainstAllPreds[j] = twoAgainstAll.predict(data[j]);
            twoAgainstAllPredsScore[j] = twoAgainstAll.score(data[j]);
            if (twoAgainstAllPreds[j] == twoAgainstAllClasses[j])
                predCorrect++;
        }
        System.out.println("\nSVM: 2 AGAINST ALL:"
                            + "\nCorrect:\t" + predCorrect
                            + "\nIncorrect:\t" + (data.length - predCorrect));
        for (int k = 0; k < data.length; k++){
            if (twoAgainstAllPreds[k] == 1){
                if (svmPreds[k] != -1){
                    if (svmPreds[k] == 0 && twoAgainstAllPredsScore[k] <= zeroAgainstAllPredsScore[k]){
                        svmPreds[k] = 0;
                    }
                    else if (svmPreds[k] == 1 && twoAgainstAllPredsScore[k] <= oneAgainstAllPredsScore[k]){
                        svmPreds[k] = 1;
                    }
                    else {
                        svmPreds[k] = 2;
                    }
                }
                else {
                    svmPreds[k] = 2;
                }
            }
        }
        
        // 3 against all
        int[] threeAgainstAllClasses = new int[classes.length];
        int[] threeAgainstAllPreds = new int[classes.length]; // predictions
        double[] threeAgainstAllPredsScore = new double[classes.length]; // score/confidence of predictions

        for (int i = 0; i < classes.length; i++){
            if (classes[i] == 3) // class = vgood
                threeAgainstAllClasses[i] = 1;
            else // class = (anything else)
                threeAgainstAllClasses[i] = -1;
        }
        LinearSVM threeAgainstAll = SVM.fit(data, threeAgainstAllClasses, new SVM.Options(softMargin));
        predCorrect = 0;
        for (int j = 0; j < data.length; j++){
            threeAgainstAllPreds[j] = threeAgainstAll.predict(data[j]);
            threeAgainstAllPredsScore[j] = threeAgainstAll.score(data[j]);
            if (threeAgainstAllPreds[j] == threeAgainstAllClasses[j])
                predCorrect++;
        }
        System.out.println("\nSVM: 3 AGAINST ALL:"
                            + "\nCorrect:\t" + predCorrect
                            + "\nIncorrect:\t" + (data.length - predCorrect));
        for (int k = 0; k < data.length; k++){
            if (threeAgainstAllPreds[k] == 1){
                if (svmPreds[k] != -1){
                    if (svmPreds[k] == 0 && threeAgainstAllPredsScore[k] <= zeroAgainstAllPredsScore[k]){
                        svmPreds[k] = 0;
                    }
                    else if (svmPreds[k] == 1 && threeAgainstAllPredsScore[k] <= oneAgainstAllPredsScore[k]){
                        svmPreds[k] = 1;
                    }
                    else if (svmPreds[k] == 2 && threeAgainstAllPredsScore[k] <= twoAgainstAllPredsScore[k]){
                        svmPreds[k] = 2;
                    }
                    else {
                        svmPreds[k] = 3;
                    }
                }
                else {
                    svmPreds[k] = 3;
                }
            }
        }
        
        // final test
        predCorrect = 0;
        for (int l = 0; l < svmPreds.length; l++){
            if (svmPreds[l] == classes[l])
                predCorrect++;
        }
        System.out.println("\nSVM (C = " + softMargin + "): FINAL:"
                            + "\nCorrect:\t" + predCorrect
                            + "\nIncorrect:\t" + (data.length - predCorrect));

    }

}
