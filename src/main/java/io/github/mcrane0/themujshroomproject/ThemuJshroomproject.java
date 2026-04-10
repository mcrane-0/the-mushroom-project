/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 */

package io.github.mcrane0.themujshroomproject;

import java.io.IOException;
import java.nio.file.FileSystems;
import java.nio.file.Path;
import smile.io.*;
import smile.data.*;
import smile.classification.SVM;

/**
 *
 * @author miles
 */
public class ThemuJshroomproject {
    
    public static void main(String[] args) throws IOException {   
        CSV carDataCSV = new CSV();
        Path pathToCSV = FileSystems.getDefault().getPath("car.data");
        DataFrame carDF = carDataCSV.read(pathToCSV);
        
        System.out.println("Hello World!");
        // System.out.println("Working Directory = " + System.getProperty("user.dir"));
        System.out.println(carDF.toString());
    }
}
