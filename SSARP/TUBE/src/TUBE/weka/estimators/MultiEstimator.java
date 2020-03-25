/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    MultiEstimator.java
 *    Copyright (C) 2009 Gabi Schmidberger
 *
 */

package weka.estimators;


import java.io.BufferedReader;
import java.io.FileReader;
import java.io.InputStreamReader;
import java.io.Reader;
import java.io.Serializable;
import java.util.Enumeration;
import java.util.Vector;

import weka.core.Capabilities;
import weka.core.CapabilitiesHandler;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Range;
import weka.core.SerializedObject;
import weka.core.Utils;
import weka.core.Capabilities.Capability;
 
/** 
 *
 <!-- globalinfo-start -->
 * Abstract class for all estimators.
 <!-- globalinfo-leftEnd -->
 *
 * Example code for a nonincremental estimator
 * <code> <pre>
 *   // create a histogram for estimation
 *   EqualWidthEstimator est = new EqualWidthEstimator();
 *   est.addValues(instances, attrIndex);
 * </pre> </code>
 *
 *
 * Example code for an incremental estimator (incremental
 * estimators must implement interface IncrementalEstimator)
 * <code> <pre>
 *   // Create a discrete estimator that takes values 0 to 9
 *   DiscreteEstimator newEst = new DiscreteEstimator(10, true);
 *
 *   // Create 50 random integers first predicting the probability of the
 *   // value, then adding the value to the estimator
 *   Random r = new Random(seed);
 *   for(int i = 0; i < 50; i++) {
 *     current = Math.abs(r.nextInt() % 10);
 *     System.out.println(newEst);
 *     System.out.println("Prediction for " + current 
 *                        + " = " + newEst.getProbability(current));
 *     newEst.addValue(current, 1);
 *   }
 * </pre> </code>
 *
 *
 * Example code for a main method for an estimator.<p>
 * <code> <pre>
 * public static void main(String [] argv) {
 *
 *   try {
 *     TUBEstimator est = new TUBEstimator();      
 *     Estimator.buildEstimator((Estimator) est, argv, false);      
 *     System.out.println(est.toString());
 *   } catch (Exception ex) {
 *     ex.printStackTrace();
 *     System.out.println(ex.getMessage());
 *   }
 * }
 * </pre> </code>
 *
 *
 * @author Gabi Schmidberger (gabi dot schmidberger at gmail dot com)
 * @author Len Trigg (trigg@cs.waikato.ac.nz)
 * @version $Revision: 1.0 $
 */
public abstract class MultiEstimator 
implements Cloneable, Serializable, OptionHandler, CapabilitiesHandler{
  
  /** for serialization */
  static final long serialVersionUID = -5902411487362274342L;
  
  /** Debugging mode */
  private boolean m_Debug = false;

  /** The attribute index of the attribute ignored (in addition to the class) */
  protected int m_notUsedAttribute = -1;
  
  
  /** The attribute index of the class attribute*/
  protected int m_classIndex = -1;
  
  /** The class value index is > -1 if subset is taken with specific class value only*/
  protected double m_classValueIndex = -1.0;
  
  /** set if class is not important */
  private boolean m_noClass = true;

  /** list of attributes that should be ignored, class would be one of them */
  Range m_ignoreList = null;

  /** The minimalvalue */
  protected double [] m_MINValue;

  /** The maximalvalue */
  protected double [] m_MAXValue;
  
  /** The rangesof the attributes */
  protected double [] m_range;
  
 /*
   * Class to support a building process of an estimator.
   */
  private static class Builder implements Serializable {
    
    /** instances of the builder */
    Instances m_instances = null;
    
    /** attribute index of the builder */
    int m_attrIndex = -1;
    
    /** class index of the builder, only relevant if class value index is set*/
    int m_classIndex = -1; 

    /** class value index of the builder */
    int m_classValueIndex = -1; 
  }
  
  /**
   * Add a new data value to the current estimator.
   *
   * @param inst the new instance to be added
   * @param weight the weight assigned to the data value 
   */
  public void addValue(Instance inst, double weight) {
    try {  
      throw new Exception("Method to add single value is not implemented!\n"+
			  "Estimator should implement MultiIncrementalEstimator.");
    } catch (Exception ex) {
      ex.printStackTrace();
      System.out.println(ex.getMessage());
    }
  }

  /**
   * Initialize the estimator with a new dataset.
   * Finds min and max first.
   *
   * @param data the dataset used to build this estimator 
   * @exception if building of estimator goes wrong
   */
  public void addValues(Instances data) throws Exception {
    // can estimator handle the data?
    getCapabilities().testWithFail(data);
   
    int classIndex = -1;
    int classValue = -1;
    addValues(data, classIndex, classValue);   
  }
  
   /**
   * Initialize the estimator using only the instance of one class. 
   * It is using the values of one attribute only.
   *
   * @param data the dataset used to build this estimator 
   * @param attrIndex attribute the estimator is for
   * @param classIndex index of the class attribute
   * @param classValue the class value 
   * @exception if building of estimator goes wrong
   */
  public void addValues(Instances data, int classIndex, int classValue) 
  throws Exception{
    // initialize ignore attribute ranges
    if (m_ignoreList == null) {
      initializeIgnoreList(data);
    }
    
    // deal with class value
    if (classIndex < 0) {
      m_noClass = true;    
    } else {
      setIgnore(classIndex);
      m_noClass = false;    
   }
    //dbo.dpln(D_TRACE, "addValues class index "+classIndex+" value "+classValue);
   
    // can estimator handle the data?
    getCapabilities().testWithFail(data);
    
    /** find the minimal and the maximal values */
    int numAttr = data.numAttributes();
    double [] min = new double[numAttr];
    double [] max = new double[numAttr];
    double [] range = new double[numAttr];
 
    double []minMax = new double[2];
    
    for (int i = 0; i < numAttr; i++) {
      if (i != classIndex) {
        
        try {
          EstimatorUtils.getMinMax(data, i, minMax);
        } catch (Exception ex) {
          ex.printStackTrace();
          System.out.println(ex.getMessage());
        }
        
        min[i] = minMax[0];
        max[i] = minMax[1];
        range[i] = max[i] - min[i];
      }
    }
    m_MINValue = min;
    m_MAXValue = max;
    m_range = range;
      
    // extract the instances with the given class value
    if (classValue >= 0) {
      Instances workData = new Instances(data, 0);
      double factor = getInstancesFromClass(data, classIndex, 
          (double)classValue, workData);
      
      // if no data return
      if (workData.numInstances() == 0) return;
      
      //addValues(workData, min, max, factor);
      addValues(workData, factor);
      
    } else {
      addValues(data, 1.0);
    }     
  }

  /**
   * Initialize the estimator using only the instance of one class. 
   * It is using the values of one attribute only.
   *
   * @param data the dataset used to build this estimator 
   * @param attrIndex attribute the estimator is for
   * @param classIndex index of the class attribute
   * @param classValue the class value 
   * @exception if building of estimator goes wrong
   */
  public void addValues(Instances data, int classIndex, double [] minValues, double [] maxValues) 
  throws Exception{
    // initialize ignore attribute ranges
    if (m_ignoreList == null) {
      initializeIgnoreList(data);
    }
    
    // deal with class value
    if (classIndex < 0) {
      m_noClass = true;    
    } else {
      setIgnore(classIndex);
      m_noClass = false;    
   }
    //dbo.dpln(D_TRACE, "addValues class index "+classIndex+" value "+classValue);
   
    // can estimator handle the data?
    getCapabilities().testWithFail(data);
    
    /** find the minimal and the maximal values */
    int numAttr = data.numAttributes();
    double [] range = new double[numAttr];
    
    for (int i = 0; i < numAttr; i++) {
      if (i != classIndex) {     
         range[i] = maxValues[i] - minValues[i];
      }
    }
    
    m_MINValue = minValues;
    m_MAXValue = maxValues;
    m_range = range;
      
    // extract the instances with the given class value
    double factor = 1.0;
    addValues(data, 1.0);
         
  }

  /**
   * Initialize the estimator with all values of one attribute of a dataset. 
   * Some estimator might ignore the min and max values.
   *
   * @param data the dataset used to build this estimator 
   * @param min minimal borders of range
   * @param max maximal borders of range
   * @param factor number of instances has been reduced to that factor
   * @exception if building of estimator goes wrong
   */
  protected void addValues(Instances data, double factor) throws Exception {

    m_classIndex = data.classIndex();
    
    // no handling of factor, would have to be overridden

    // no handling of min and max, would have to be overridden

    int numInst = data.numInstances();
    for (int i = 1; i < numInst; i++) {
      addValue(data.instance(i), 1.0);
    }
  }
 
  /**
   * Returns a dataset that contains all instances of a certain class value.
   *
   * @param data dataset to select the instances from
   * @param attrIndex index of the relevant attribute
   * @param classIndex index of the class attribute
   * @param classValue the relevant class value 
   * @return a dataset with only 
   */
  private double getInstancesFromClass(Instances data, int classIndex,
      double classValue, Instances workData) {
    //DBO.pln("getInstancesFromClass classValue"+classValue+" workData"+data.numInstances());
    
    int num = 0;
    int numClassValue = 0;
    for (int i = 0; i < data.numInstances(); i++) {
      num++;
      if (data.instance(i).value(classIndex) == classValue) {
        workData.add(data.instance(i));
        numClassValue++;
      }     
    } 
    
    Double alphaFactor = new Double((double)numClassValue/(double)num);
    return alphaFactor;
  }

  /**
   * Switches the attributes that are on this list to be ignored
   * 
   * @param list list of integers, all are used for an output type
   */
  public void initializeIgnoreList(Instances data) {
    if (data != null) {
      // initialize ignore attribute ranges
      m_ignoreList = new Range();
      m_ignoreList.setUpper(data.numAttributes());     
    }
  }
  
 /**
   * Switches the attributes that are on this list to be ignored
   * 
   * @param list list of integers, all are used for an output type
   */
  public void setIgnoreList(String list) {
    if (list.length() > 0) {
       m_ignoreList.setRanges(list);
     }
  }
  
  public void setIgnore(int att) {
    if (att >= 0) {
      String attrList = m_ignoreList.getRanges() + ", "+att;
      m_ignoreList.setRanges(attrList);
    }
  }

  /**
   * Get a probability estimate for a value vector (instance)
   *
   * @param inst the vector of values to compute the prob off
   * @return the estimated probability of the supplied value
   */
  public abstract double getProbability(Instance inst, double classIndex);

  /**
   * Build an estimator using the options. The data is given in the options.
   *
   * @param est the estimator used
   * @param options the list of options
   * @param isIncremental true if estimator is incremental
   * @exception Exception if something goes wrong or the user requests help on
   * command options
   */
  public static void buildEstimator(MultiEstimator est, String [] options,
				    boolean isIncremental) 
    throws Exception {
    //DBO.pln("buildEstimator");
    
    boolean debug = false;
    boolean helpRequest;
    
    // read all options
    Builder build = new Builder();
    try {
      setGeneralOptions(build, est, options);
      
      if (est instanceof OptionHandler) {
	((OptionHandler)est).setOptions(options);
      }
      
      Utils.checkForRemainingOptions(options);
      
    
      buildEstimator(est, build.m_instances, build.m_attrIndex,
		     build.m_classIndex, build.m_classValueIndex, isIncremental);
    } catch (Exception ex) {
      ex.printStackTrace();
      System.out.println(ex.getMessage());
      String specificOptions = "";
      // Output the error and also the valid options
      if (est instanceof OptionHandler) {
	specificOptions += "\nEstimator options:\n\n";
	Enumeration enumOptions = ((OptionHandler)est).listOptions();
	while (enumOptions.hasMoreElements()) {
	  Option option = (Option) enumOptions.nextElement();
	  specificOptions += option.synopsis() + '\n'
	    + option.description() + "\n";
	}
      }
      
      String genericOptions = "\nGeneral options:\n\n"
	+ "-h\n"
	+ "\tGet help on available options.\n"
	+ "-i <file>\n"
	+ "\tThe name of the file containing input instances.\n"
	+ "\tIf not supplied then instances will be read from stdin.\n"
	+ "-a <attribute index>\n"
	+ "\tThe number of the attribute the probability distribution\n"
	+ "\testimation is done for.\n"
	+ "\t\"first\" and \"last\" are also valid entries.\n"
	+ "\tIf not supplied then no class is assigned.\n"
	+ "-c <class index>\n"
	+ "\tIf class value index is set, this attribute is taken as class.\n"
	+ "\t\"first\" and \"last\" are also valid entries.\n"
	+ "\tIf not supplied then last is default.\n"
	+ "-v <class value index>\n"
	+ "\tIf value is different to -1, select instances of this class value.\n"
	+ "\t\"first\" and \"last\" are also valid entries.\n"
	+ "\tIf not supplied then all instances are taken.\n";
      
      throw new Exception('\n' + ex.getMessage()
			  + specificOptions+genericOptions);
    }
  }

  public static void buildEstimator(MultiEstimator est,
      Instances instances, int attrIndex, 
      int classIndex, int classValueIndex,
      boolean isIncremental) throws Exception {
    
    // DBO.pln("buildEstimator 2 " + classValueIndex);
    
    // non-incremental estimator add all instances at once
    if (!isIncremental) {
      
      if (classValueIndex == -1) {
        // DBO.pln("before addValues -- Estimator");
        est.addValues(instances);
      } else {
        // DBO.pln("before addValues with classvalue -- Estimator");
        est.addValues(instances, classIndex, classValueIndex);
      }
    } else {
      // incremental estimator, read one value at a time
      Enumeration enumInsts = (instances).enumerateInstances();
      while (enumInsts.hasMoreElements()) {
        Instance instance = 
          (Instance) enumInsts.nextElement();
        ((MultiIncrementalEstimator)est).addValue(instance,
            instance.weight());
      }
    }
  }
  
  /**
   * Parses and sets the general options
   * @param build contains the data used
   * @param est the estimator used
   * @param options the options from the command line
   */
  private static void setGeneralOptions(Builder build, MultiEstimator est, 
      String [] options)  
  throws Exception {
    Reader input = null;
    
    // help request option
    boolean helpRequest = Utils.getFlag('h', options);
    if (helpRequest) {
      throw new Exception("Help requested.\n");
    }
    
    // instances used
    String infileName = Utils.getOption('i', options);
    if (infileName.length() != 0) {
      input = new BufferedReader(new FileReader(infileName));
    } else {
      input = new BufferedReader(new InputStreamReader(System.in));
    }
    
    build.m_instances = new Instances(input);
    
    // attribute index
    String attrIndex = Utils.getOption('a', options);
    
    if (attrIndex.length() != 0) {
      if (attrIndex.equals("first")) {
        build.m_attrIndex = 0;
      } else if (attrIndex.equals("last")) {
        build.m_attrIndex = build.m_instances.numAttributes() - 1;
      } else {
        int index = Integer.parseInt(attrIndex) - 1;
        if ((index < 0) || (index >= build.m_instances.numAttributes())) {
          throw new IllegalArgumentException("Option a: attribute index out of range.");
        }
        build.m_attrIndex = index;
        
      }
    } else {
      // default is the first attribute
      build.m_attrIndex = 0;
    }
    
    //class index, if not given is set to last attribute
    String classIndex = Utils.getOption('c', options);
    if (classIndex.length() == 0) classIndex = "last";
    
    if (classIndex.length() != 0) {
      if (classIndex.equals("first")) {
        build.m_classIndex = 0;
      } else if (classIndex.equals("last")) {
        build.m_classIndex = build.m_instances.numAttributes() - 1;
      } else {
        int cl = Integer.parseInt(classIndex);
        if (cl == -1) {
          build.m_classIndex = -1;
        } else {
          build.m_classIndex = cl - 1;	
        }
      }
      build.m_instances.setClassIndex(build.m_classIndex);
    } 
    
    //class value index, if not given is set to -1
    String classValueIndex = Utils.getOption('v', options);
    if (classValueIndex.length() != 0) {
      if (classValueIndex.equals("first")) {
        build.m_classValueIndex = 0;
      } else if (classValueIndex.equals("last")) {
        build.m_classValueIndex = build.m_instances.numAttributes() - 1;
      } else {
        int cl = Integer.parseInt(classValueIndex);
        if (cl == -1) {
          build.m_classValueIndex = -1;
        } else {
          build.m_classValueIndex = cl - 1;	
        }
      }
    } 
    
    build.m_instances.setClassIndex(build.m_classIndex);
  }
  
  /**
   * Creates a deep copy of the given estimator using serialization.
   *
   * @param model the estimator to copy
   * @return a deep copy of the estimator
   * @exception Exception if an error occurs
   */
  public static MultiEstimator clone(MultiEstimator model) throws Exception {
    
    return makeCopy(model);
  }
  
  /**
   * Creates a deep copy of the given estimator using serialization.
   *
   * @param model the estimator to copy
   * @return a deep copy of the estimator
   * @exception Exception if an error occurs
   */
  public static MultiEstimator makeCopy(MultiEstimator model) throws Exception {

    return (MultiEstimator)new SerializedObject(model).getObject();
  }

  /**
   * Creates a given number of deep copies of the given estimator using serialization.
   * 
   * @param model the estimator to copy
   * @param num the number of estimator copies to create.
   * @return an array of estimators.
   * @exception Exception if an error occurs
   */
  public static MultiEstimator [] makeCopies(MultiEstimator model,
					 int num) throws Exception {

    if (model == null) {
      throw new Exception("No model estimator set");
    }
    MultiEstimator [] estimators = new MultiEstimator [num];
    SerializedObject so = new SerializedObject(model);
    for(int i = 0; i < estimators.length; i++) {
      estimators[i] = (MultiEstimator) so.getObject();
    }
    return estimators;
  }
 
  /**
   * Tests whether the current estimation object is equal to another
   * estimation object
   *
   * @param obj the object to compare against
   * @return true if the two objects are equal
   */
  public boolean equals(Object obj) {
    
    if ((obj == null) || !(obj.getClass().equals(this.getClass()))) {
      return false;
    }
    MultiEstimator cmp = (MultiEstimator) obj;
    if (m_Debug != cmp.m_Debug) return false;
    if (m_classValueIndex != cmp.m_classValueIndex) return false;
    if (m_noClass != cmp.m_noClass) return false;
    
    return true;
  }

  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   */
  public Enumeration listOptions() {

    Vector newVector = new Vector(1);

    newVector.addElement(new Option(
	      "\tIf set, estimator is run in debug mode and\n"
	      + "\tmay output additional info to the console",
	      "D", 0, "-D"));
    return newVector.elements();
  }

  /**
   * Parses a given list of options. Valid options are:<p>
   *
   * -D  <br>
   * If set, estimator is run in debug mode and 
   * may output additional info to the console.<p>
   *
   * @param options the list of options as an array of strings
   * @exception Exception if an option is not supported
   */
  public void setOptions(String[] options) throws Exception {

    setDebug(Utils.getFlag('D', options));

    String notUsedAttribute = Utils.getOption('I', options);
    if (notUsedAttribute.length() != 0) {
      setNotUsedAttribute(Integer.parseInt(notUsedAttribute));
    }
 }

  /**
   * Gets the current settings of the Estimator.
   *
   * @return an array of strings suitable for passing to setOptions
   */
  public String [] getOptions() {

    String [] options;
    if (getDebug()) {
      options = new String[1];
      options[0] = "-D";
    } else {
      options = new String[0];
    }
    return options;
  }
  
  /**
   * Creates a new instance of a estimatorr given it's class name and
   * (optional) arguments to pass to it's setOptions method. If the
   * classifier implements OptionHandler and the options parameter is
   * non-null, the classifier will have it's options set.
   *
   * @param name the fully qualified class name of the estimatorr
   * @param options an array of options suitable for passing to setOptions. May
   * be null.
   * @return the newly created classifier, ready for use.
   * @exception Exception if the classifier name is invalid, or the options
   * supplied are not acceptable to the classifier
   */
  public static MultiEstimator forName(String name,
      String [] options) throws Exception {
    
    return (MultiEstimator)Utils.forName(MultiEstimator.class,
        name,
        options);
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String debugTipText() {
    return "If set to true, estimator may output additional info to " +
      "the console.";
  }
 
 /**
   * Set debugging mode.
   *
   * @param debug true if debug output should be printed
   */
  public void setDebug(boolean debug) {

    m_Debug = debug;
  }

  /**
   * Get whether debugging is turned on.
   *
   * @return true if debugging output is on
   */
  public boolean getDebug() {

    return m_Debug;
  }
  
  /**
   * Sets the maximum number of bins the range can be split into
   *
   * @param notUsedAttribute the index of the not used attribute 
  */
  public void setNotUsedAttribute(int att) {
    m_notUsedAttribute = att;
  }
  
  /**
   * Returns the maximum number of bins 
   *
   * @return notUsedAttribute the index of the not used attribute 
   */
  public int getNotUsedAttribute() {
    return m_notUsedAttribute;
  }
  
 /**
   * Get the minimal value of this estimator
   * @param attrIndex the index of the attribute
   * @return the minimal value of an attribute
   */
  public double getMINValue(int attrIndex) {

    return m_MINValue[attrIndex];
  }

  /**
   * Get the maximal value of this estimator
   * @param attrIndex the index of the attribute
   * @return the maximal value of an attribute
   */
  public double getMAXValue(int attrIndex) {

    return m_MAXValue[attrIndex];
  }

  /** 
   * Returns the Capabilities of this Estimator. Derived estimators have to
   * override this method to enable capabilities.
   *
   * @return            the capabilities of this object
   * @see               Capabilities
   */
  public Capabilities getCapabilities() {
    Capabilities result = new Capabilities(this);
    
    result.enable(Capability.NUMERIC_ATTRIBUTES);
    result.enable(Capability.NOMINAL_ATTRIBUTES);

    // class
    if (!m_noClass) {
      result.enable(Capability.NOMINAL_CLASS);
      result.enable(Capability.MISSING_CLASS_VALUES);
    } else {
      result.enable(Capability.NO_CLASS);
    }
       
    return result;
  }
  
  /** 
   * Test if the estimator can handle the data.
   * @param data the dataset the estimator takes an attribute from
   * @param attrIndex the index of the attribute
   * @see Capabilities
   */
  public void testCapabilities(Instances data, int attrIndex) throws Exception {
    getCapabilities().testWithFail(data);
    getCapabilities().testWithFail(data.attribute(attrIndex));
  }
}








