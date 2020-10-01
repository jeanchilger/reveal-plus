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
 *    NaiveBayesParametrized.java
 *    Copyright (C) 2009 University of Waikato
 *
 */

package weka.classifiers.bayes;

import java.util.Enumeration;
import java.util.Vector;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.AdditionalMeasureProducer;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.RevisionUtils;
import weka.core.UnsupportedAttributeTypeException;
import weka.core.UnsupportedClassTypeException;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.core.Capabilities.Capability;
import weka.core.Debug.DBO;
import weka.estimators.BinningEstimator;
import weka.estimators.BinningUtils;
import weka.estimators.DiscreteEstimator;
import weka.estimators.Estimator;
import weka.estimators.IncrementalEstimator;
import weka.estimators.TUBEstimator;
import weka.filters.Filter;


/**
 <!-- globalinfo-start -->
 * Class for a Naive Bayes classifier with parameter for estimator. An estimator for numeric data and a discretizing filter can be given to this class per parameter.<br/>
 * The discretizer is transforming numeric attributes to nominal attributes before building the model.<br/>
 * This classifier is not an UpdateableClassifier.
 <!-- globalinfo-leftEnd -->
 *
 <!-- options-start -->

 * Valid options are:<p/>
 *
 *  <pre> -Z <filter specification> <br>
 * Full class name of discretizing filter to use, followed
 * by filter options. <p>
 * eg: "weka.filters.unsupervised.attribute.Discretize -B 30"  <p>
 *
 *  <pre> -E <estimator specification> <br>
 * Full class name of estimator to use, followed 
 * by estimator options.<p>
 * eg: "weka.estimators.NormalEstimator"
 *
 *  <pre> -V <option list> <br>
 * Switch on verbose mode and give list of output options
 * eg: 1,2,11.
 *
 <!-- options-leftEnd -->
 *
 * @author Gabi Schmidberger (gabi dot schmidberger at gmail dot com)
 * contains code copied from Naive Bayes.java
 * @author Len Trigg (trigg@cs.waikato.ac.nz)
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision: 1.0 $
 */
public class NaiveBayesParametrized extends Classifier 
  implements OptionHandler, WeightedInstancesHandler,
   AdditionalMeasureProducer
{

  /** for serialization */
  static final long serialVersionUID = -8874594146976212939L;

  /** additional output element (for verbose modes) */
  DBO dbo = new DBO();

  /** output probabilities */
  public static int D_ALL_PROBS   = 0; // 1 on the command line

  /** output of how many cuts */
  public static int D_ILLCUTS     = 1; // 2 

  /** Print a estimation curve for each estimator */
  public static int D_PR_CURVES   = 2; // 3

  /** Print if filter was used and data format (in any case) */
  public static int D_DATAFORMAT  = 3; // 4

  /** The attribute estimators. */
  protected Estimator [][] m_Distributions;
  
  /** The class estimator. */
  protected Estimator m_ClassDistribution;

  /** The estimator set by option (only with E option) */
  protected Estimator m_estimator = new TUBEstimator();

  /** The discretizing filter set by option  */
  protected Filter m_discretizer = new weka.filters.supervised.attribute.Discretize();

  /** The options string for the LoglikelihoodEstimator */
  String [] m_LLOptions = null;

  /**
   * Whether to use discretization than normal distribution
   * for numeric attributes
   */
  protected boolean m_UseDiscretizer = false;

  /** The number of classes (or 1 for numeric class) */
  protected int m_NumClasses;

  /**
   * The dataset header for the purposes of printing out a semi-intelligible 
   * model 
   */
  protected Instances m_Instances;

  /*** The precision parameter used for numeric attributes */
  protected static final double DEFAULT_NUM_PRECISION = 0.01;

  /** number of illegal cuts in CV in average */
  double m_avgCVIllegalCuts = 0.0;

  /** number of illegal cuts in final model */
  double m_numIllegalCuts = 0.0;

  /** difference between the number of illegal cuts in CV and leftEnd model */
  double m_diffNumIllegalCuts = 0.0;


  /** Constructor */
  public NaiveBayesParametrized() {
    // for the debug output
    dbo.initializeRanges(15);
  }

  /**
   * Returns a string describing this classifier
   * @return a description of the classifier suitable for
   * displaying in the explorer/experimenter gui
   */
  public String globalInfo() {
    return "Class for a Naive Bayes classifier. An estimator for numeric data"
      +" and a discretizing filter can be given to this class per parameter.\n"
      +" The discretizer is transforming numeric attributes to nominal"
      +" attributes before building the model.\n"
      +" This classifier is not an UpdateableClassifier.";
  }

  /**
   * Returns default capabilities of the classifier.
   *
   * @return      the capabilities of this classifier
   */
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();

    // attributes
    result.enable(Capability.NOMINAL_ATTRIBUTES);
    result.enable(Capability.NUMERIC_ATTRIBUTES);
    result.enable(Capability.MISSING_VALUES);

    // class
    result.enable(Capability.NOMINAL_CLASS);
    result.enable(Capability.MISSING_CLASS_VALUES);

    // instances
    result.setMinimumNumberInstances(0);
    
    return result;
  }

  /**
   * Returns an enumeration of the additional measure names.
   *
   * @return an enumeration of the measure names
   */
  public Enumeration enumerateMeasures() {
    
    Vector newVector = new Vector(3);
    newVector.addElement("measureIllegalCuts");
    newVector.addElement("measureDiffIllegalCuts");
    newVector.addElement("measureAvgCVIllegalCuts");
    //newVector.addElement("");
    return newVector.elements();
  }
  
  /**
   * Returns the value of the named measure.
   *
   * @param measureName the name of the measure to query for its value
   * @return the value of the named measure
   * @exception IllegalArgumentException if the named measure is not supported
   */
  public double getMeasure(String additionalMeasureName) {
    
    
    if (additionalMeasureName.equalsIgnoreCase("measureIllegalCuts")) {
      return measureIllegalCuts();
    }
    else if (additionalMeasureName.equalsIgnoreCase("measureDiffIllegalCuts")) {
      return measureDiffIllegalCuts();
    }
    else if (additionalMeasureName.equalsIgnoreCase("measureAvgCVIllegalCuts")) {
      return measureAvgCVIllegalCuts();
    }
    //    else if (additionalMeasureName.equalsIgnoreCase("measureNumLeaves")) {
    //  return measureNumLeaves();
    else {throw new IllegalArgumentException(additionalMeasureName 
					     + " not supported (TUBEstimator)");
    }
  }
  
  /**
   * Measure function for average of CV illegal cuts.
   *
   * @return the average of the average of CV illegal cuts
   */
  public double measureAvgCVIllegalCuts() {
    
    return m_avgCVIllegalCuts;
  }
  
  /**
   * Measure function for illegalCuts.
   *
   * @return the number of illegal cuts in the average of models
   */
  public double measureIllegalCuts() {
    
    return m_numIllegalCuts;
  }
  
  /**
   * Measure function for difference of illegalCuts.
   *
   * @return the difference of illegalCuts
   */
  public double measureDiffIllegalCuts() {
    
    return m_diffNumIllegalCuts;
  }

  /**
   * Return a copy of the bins of one of the estimators if it is a DiscretizingEstimator
   *
   * @param attrIndex the attribute index
   * @param classIndex the classes index
   * @return a copy of the bins
   */
  public Vector getBinsCopy (int attrIndex, int classIndex) {
    Vector bins = null;
    Estimator est = m_Distributions[attrIndex][classIndex];
    if (est instanceof BinningEstimator) {
      bins = ((BinningEstimator)est).getBinsCopy();
    }
    return bins;
  }  

  /**
   * Generates the classifier.
   *
   * @param instances set of instances serving as training data 
   * @exception Exception if the classifier has not been generated 
   * successfully
   */
  public void buildClassifier(Instances instances) throws Exception {

    // can classifier handle the data?
    getCapabilities().testWithFail(instances);

    if (getDebug()) {
      DBO.pln("debug is set on");
      dbo.setVerboseOn();
    }
    dbo.dpln("verbose is set on");

    if (instances.checkForStringAttributes()) {
      throw new UnsupportedAttributeTypeException("Cannot handle string attributes!");
    }
    if (instances.classAttribute().isNumeric()) {
      throw new UnsupportedClassTypeException("Naive Bayes: Class is numeric!");
    }
    m_NumClasses = instances.numClasses();
    if (m_NumClasses < 0) {
      throw new Exception ("Dataset has no class attribute");
    }
    
    // Copy the instances
    m_Instances = new Instances(instances);
    
    // Discretize instances if required
    if (m_UseDiscretizer) {
      if (m_discretizer == null) {
	m_discretizer = new weka.filters.supervised.attribute.Discretize();
      }
      m_discretizer.setInputFormat(m_Instances);
      m_Instances = weka.filters.Filter.useFilter(m_Instances, m_discretizer);

      dbo.dpln(D_DATAFORMAT, "filter was used: "+getDiscretizerSpec());
      
      
    } else {
      m_discretizer = null;
    }
   
    if (dbo.dl(D_DATAFORMAT)) {
      Instances d = new Instances(m_Instances, 0);
      dbo.dpln(D_DATAFORMAT, "current Dataformat is"+d);
    }
 
    // Reserve space for the distributions
    m_Distributions = new Estimator[m_Instances.numAttributes() - 1]
      [m_Instances.numClasses()];
    m_ClassDistribution = new DiscreteEstimator(m_Instances.numClasses(), 
						true);
    int attIndex = 0;
    Enumeration enumAtts = m_Instances.enumerateAttributes();
    while (enumAtts.hasMoreElements()) {

      Attribute attribute = (Attribute) enumAtts.nextElement();
      
      // If the attribute is numeric, determine the estimator 
      // numeric precision from differences between adjacent values
      double numPrecision = DEFAULT_NUM_PRECISION;
      if (attribute.type() == Attribute.NUMERIC) {
	m_Instances.sort(attribute);
	if ((m_Instances.numInstances() > 0)
	    && !m_Instances.instance(0).isMissing(attribute)) {
	  double lastVal = m_Instances.instance(0).value(attribute);
	  double currentVal, deltaSum = 0;
	  int distinct = 0;
	  for (int i = 1; i < m_Instances.numInstances(); i++) {
	    Instance currentInst = m_Instances.instance(i);
	    if (currentInst.isMissing(attribute)) {
	      break;
	    }
	    currentVal = currentInst.value(attribute);
	    if (currentVal != lastVal) {
	      deltaSum += currentVal - lastVal;
	      lastVal = currentVal;
	      distinct++;
	    }
	  }
	  if (distinct > 0) {
	    numPrecision = deltaSum / distinct;
	  }
	}
      }
      for (int j = 0; j < m_Instances.numClasses(); j++) {
	switch (attribute.type()) {
	case Attribute.NUMERIC: 
	  m_Distributions[attIndex][j] = 
	    Estimator.makeCopy((Estimator)m_estimator);
	  break;
	case Attribute.NOMINAL:
	  m_Distributions[attIndex][j] = 
	    new DiscreteEstimator(attribute.numValues(), true);
	  break;
	default:
	  throw new Exception("Attribute type unknown to NaiveBayes");
	}
      }
      
      attIndex++;
    }
  
    // build estimators
    Enumeration enumInsts = m_Instances.enumerateInstances();
    while (enumInsts.hasMoreElements()) {
      Instance instance = 
	(Instance) enumInsts.nextElement();
      updateClassifier(instance);
    }
    
    // build the nonincremental estimators
    buildNonincrementalEstimators(m_Instances);    

    // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
    // output density curves
    if (dbo.dl(D_PR_CURVES)) {
      String filename = m_Instances.relationName();
      if (filename.length() > 20) {
	filename = (filename.substring(0, 20));
      }
      
      int attrIndex = 0;
      enumAtts = m_Instances.enumerateAttributes();
      while (enumAtts.hasMoreElements()) {
	// -----------------------------
	// get min and max for writeCurve    
	// find min and max 
	double [] minMax = new double[2];
	
	try {
	  BinningUtils.getMinMax(m_Instances, attrIndex, minMax);
	} catch (Exception ex) {
	  ex.printStackTrace();
	  System.out.println(ex.getMessage());
	}
	double minValue = minMax[0];
	double maxValue = minMax[1];
	// ------------------------------
	
	Attribute attribute = (Attribute) enumAtts.nextElement();
	
	// print the estimator curve
	if (attribute.type() == Attribute.NUMERIC) {
	  m_Instances.sort(attribute);
	  for (int j = 0; j < m_Instances.numClasses(); j++) {
	    String str = null;
	    str = "E";
	    
	    BinningUtils.writeCurve(filename+"-"+attrIndex+"-"+j+str, 
				      m_Distributions[attrIndex][j],
				      m_ClassDistribution, (double)j,
				      minValue, maxValue, 400);
	  }
	}
	attrIndex++;
      }
    }
    // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 

    // Save space
    m_Instances = new Instances(m_Instances, 0);
  }


  /**
   * Updates the classifier with the given instance.
   *
   * @param instance the new training instance to include in the model 
   * @exception Exception if the instance could not be incorporated in
   * the model.
   */
  public void updateClassifier(Instance instance) throws Exception {
    
    if (!instance.classIsMissing()) {
      Enumeration enumAtts = m_Instances.enumerateAttributes();
      int attIndex = 0;
      while (enumAtts.hasMoreElements()) {
	Attribute attribute = (Attribute) enumAtts.nextElement();
	if (!(instance.isMissing(attribute)) &&
	    ((m_Distributions[attIndex][(int)instance.classValue()] instanceof 
	    IncrementalEstimator))) {
	  ((IncrementalEstimator)
	   m_Distributions[attIndex][(int)instance.classValue()]).
	    addValue(instance.value(attribute), instance.weight());
	}
	attIndex++;
      }
      ((DiscreteEstimator)m_ClassDistribution).addValue(instance.classValue(),
		  instance.weight());
    }
  }

  /**
   * Updates the classifier with the given instance.
   *
   * @param instances the dataset to build the estimators with 
   */
  public void buildNonincrementalEstimators(Instances instances)  throws Exception {

    double alpha = Double.NaN;

    // get class probabilities to compute alphas
    double [] probs = new double[m_NumClasses];
    for (int classValue = 0; classValue < m_NumClasses; classValue++) {
      probs[classValue] = m_ClassDistribution.getProbability(classValue);
    }

    // set alphas and numInstforillcut -  int attIndex = 0;
    int attIndex = 0;
    for (int i = 0; i < instances.numAttributes(); i++) {
      if (attIndex != instances.classIndex()) {
	for (int classValue = 0; 
	     classValue < instances.numClasses(); classValue++) {
	  if (m_Distributions[attIndex][classValue] instanceof BinningEstimator) {
	    if (Double.isNaN(alpha)) {
	      alpha = ((BinningEstimator)m_Distributions[attIndex][classValue]).getAlpha();
	    }
	    double p = probs[classValue];
	    ((BinningEstimator)m_Distributions[attIndex][classValue]).setAlpha(alpha * probs[classValue]);
	    ((BinningEstimator)m_Distributions[attIndex][classValue]).
	      setNumInstForIllCut((double)instances.numInstances());
	  }
	}
      }
    }

    // build estimators - for each attribute - for each class
    int numModels = 0;   
    attIndex = 0;
    for (int i = 0; i < instances.numAttributes(); i++) {
      if (attIndex != instances.classIndex()) {
	for (int classValue = 0; 
	     classValue < instances.numClasses(); classValue++) {
	  if (!(m_Distributions[attIndex][classValue] instanceof IncrementalEstimator) 
	      && !(m_Distributions[attIndex][classValue] instanceof DiscreteEstimator)) {
	    ((Estimator)m_Distributions[attIndex][classValue]).addValues(instances,
									 attIndex, 
									 instances.classIndex(),
									 classValue);
	    numModels++;
	    // gather illegal cut data
	    m_numIllegalCuts += ((BinningEstimator)m_Distributions[attIndex][classValue]).
	      getNumIllegalCuts();
	    dbo.dpln(D_ILLCUTS, "numIllegalCuts "+m_numIllegalCuts);
	    m_avgCVIllegalCuts += ((BinningEstimator)m_Distributions[attIndex][classValue]).
	      getAvgCVIllegalCuts();
	    m_diffNumIllegalCuts += ((BinningEstimator)m_Distributions[attIndex][classValue]).
	      getDiffNumIllegalCuts();
	  }
	}
	
	attIndex++; //should be here not 2 below
      }
    }
    if (numModels > 0) { 
      m_numIllegalCuts = m_numIllegalCuts / numModels;
      dbo.dpln(D_ILLCUTS, "AVG numIllegalCuts "+m_numIllegalCuts);
      m_avgCVIllegalCuts = m_avgCVIllegalCuts / numModels;
      m_diffNumIllegalCuts = m_diffNumIllegalCuts / numModels;
    }
  }

  /**
   * Calculates the class membership probabilities for the given test 
   * instance.
   *
   * @param instance the instance to be classified
   * @return predicted class probability distribution
   * @exception Exception if there is a problem generating the prediction
   */
  public double [] distributionForInstance(Instance instance) 
  throws Exception { 
    dbo.dpln(D_ALL_PROBS, "distributionForInstance "+ instance);
    
    if (m_UseDiscretizer) {
      m_discretizer.input(instance);
      instance = m_discretizer.output();
    }

    // probability for each class
    double [] probs = new double[m_NumClasses];
    double max = 0;
    // used if out of border should be set unclassified
    //double [] prtest = new double[m_NumClasses];

    // initialize with the probability of the class
    for (int classIndex = 0; classIndex < m_NumClasses; classIndex++) {
      probs[classIndex] = m_ClassDistribution.getProbability(classIndex);
      //prtest[classIndex] = 0.0;
    }

    Enumeration enumAtts = instance.enumerateAttributes();
    int attIndex = 0;
    while (enumAtts.hasMoreElements()) {
      Attribute attribute = (Attribute) enumAtts.nextElement();
      if (!instance.isMissing(attribute)) {
	double temp = 0;
	max = 0;
	for (int classIndex = 0; classIndex < m_NumClasses; classIndex++) {

	  temp = Math.max(1e-75, m_Distributions[attIndex][classIndex].
			  getProbability(instance.value(attribute)));
	  // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
	  //prtest[classIndex] += m_Distributions[attIndex][classIndex].
	  //		  getProbability(instance.value(attribute));


	  dbo.dpln(D_ALL_PROBS, "probability  "+temp+" forclass "+classIndex +" foratt "+attIndex);

	  probs[classIndex] *= temp;
	   if (dbo.dl(D_ALL_PROBS)) {
	      DBO.p("probs[]");
	      for (int cl = 0; cl < m_NumClasses; cl++) {
		DBO.p(" "+probs[cl]);
		
	      }
	      DBO.pln("");
	    }
	  if (probs[classIndex] > max) {
	    max = probs[classIndex];
	    if ((max > 0) && (max < 1e-25)) { // Danger of probability underflow
	      for (int cl = 0; cl < m_NumClasses; cl++) {
		probs[cl] *= 1e25;
	      }
	    }
	    if ((max > 0) && (max > 1e50)) { // Danger of probability overflow
	      for (int cl = 0; cl < m_NumClasses; cl++) {
		probs[cl] *= 1e-50;
	      }
	    }
	  }
	  if (Double.isNaN(probs[classIndex])) {
	    throw new Exception("NaN returned from estimator for attribute "
		+ attribute.name() + ":\n"
		+ m_Distributions[attIndex][classIndex].toString());
	  }
	}

      }
      attIndex++;
    }

    // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
    if (dbo.dl(D_ALL_PROBS)) {
      DBO.pln(" ******** ");
      ///DBO.pln(""+instance);
      DBO.p("probs[]");
      for (int classIndex = 0; classIndex < m_NumClasses; classIndex++) {
	DBO.p(" "+probs[classIndex]);
	
      }
      DBO.pln("");
    }
    // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 

    // Display probabilities
    Utils.normalize(probs);

    // all probabilities have been 0.0 because falling out of range
    // set resulting probability to 0.0
//     for (int classIndex = 0; classIndex < m_NumClasses; classIndex++) {
//       if (prtest[classIndex] == 0.0) { 
// 	probs[classIndex] = 0.0;
// 	dbo.pln("UNCLASSIFIED!");
//       }
//     }

    dbo.dpln(D_ALL_PROBS, "***");
    return probs;
  }

  /**
   * Calculates the differences between the probabilities
   *
   * @param valueReal the real class value of the instance
   * @param valueFound the classified class value
   * @param instance the instance to be classified
   * @return difference valuefound - valuereal
   * @exception Exception if there is a problem generating the prediction
   */
  public double diffClassProbs(double valueReal, double valueFound,
			       Instance instance) {

    double clProbReal = m_ClassDistribution.getProbability(valueReal);
    double clProbFound = m_ClassDistribution.getProbability(valueFound);

    Enumeration enumAtts = instance.enumerateAttributes();
    int attIndex = 0;
    while (enumAtts.hasMoreElements()) {
      Attribute attribute = (Attribute) enumAtts.nextElement();
      if (!instance.isMissing(attribute)) {
	double temp = 0;
	temp = Math.max(1e-75, m_Distributions[attIndex][(int) valueReal].
			getProbability(instance.value(attribute)));
	clProbReal *= temp;
      } 
    }
    attIndex++;
    
    enumAtts = instance.enumerateAttributes();
    attIndex = 0;
    while (enumAtts.hasMoreElements()) {
      Attribute attribute = (Attribute) enumAtts.nextElement();
      if (!instance.isMissing(attribute)) {
	double temp = 0;
	temp = Math.max(1e-75, m_Distributions[attIndex][(int) valueFound].
			getProbability(instance.value(attribute)));
	clProbFound *= temp;
      } 
    }
    attIndex++;
    
    return clProbFound - clProbReal;
  }

  /**
   * Calculates the class membership probabilities for the given test 
   * instance.
   *
   * @param instance the instance to be classified
   * @return predicted class probability distribution
   * @exception Exception if there is a problem generating the prediction
   */
  public Instance  makeIndInstance(Instances inst, Instance instance,
				   int classValue, Instance newInstance) 
  throws Exception {

    if (m_UseDiscretizer ) {
      m_discretizer.input(instance);
      instance = m_discretizer.output();
    }
    double [] probs = new double[m_NumClasses];
    for (int j = 0; j < m_NumClasses; j++) {
      probs[j] = m_ClassDistribution.getProbability(j);
    }
    Enumeration enumAtts = instance.enumerateAttributes();
    int attIndex = 0;
    while (enumAtts.hasMoreElements()) {
      Attribute attribute = (Attribute) enumAtts.nextElement();
      if (!instance.isMissing(attribute)) {
	double temp, max = 0;
	double dist [] = new double[m_NumClasses];
	for (int j = 0; j < m_NumClasses; j++) {
	  temp = Math.max(1e-75, m_Distributions[attIndex][j].
			  getProbability(instance.value(attribute)));
	  dist[j] = Math.max(1e-75, m_Distributions[attIndex][j].
			     getProbability(instance.value(attribute)));
	  probs[j] *= temp;
	  if (probs[j] > max) {
	    max = probs[j];
	  }
	  if (Double.isNaN(probs[j])) {
	    throw new Exception("NaN returned from estimator for attribute "
				+ attribute.name() + ":\n"
				+ m_Distributions[attIndex][j].toString());
	  }
	}
	// Which class would this attribute have chosen
	int pred = Utils.maxIndex(dist);
	if (pred == classValue) {
	  //DBO.p("-miss-");
	  newInstance.setMissing(attIndex);
	}
	//DBO.pln(""+pred);
	
	if ((max > 0) && (max < 1e-75)) { // Danger of probability underflow
	  for (int j = 0; j < m_NumClasses; j++) {
	    probs[j] *= 1e75;
	  }
	}
      }
      attIndex++;
    }
    return newInstance;
  }

  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   */
  public Enumeration listOptions() {

    Vector newVector = new Vector(4);
    newVector.addElement(new Option(
	      "\tFull class name of discretizing filter to use, followed\n"
	      + "\tby filter options.\n"
	      + "\teg: \"weka.filters.unsupervised.attribute.Discretize -B 30\"",
	      "Z", 1, "-Z <filter specification>"));

    newVector.addElement(new Option(
	      "\tFull class name of estimator to use, followed\n"
	      + "\tby estimator options.\n"
	      + "\teg: \"weka.estimators.NormalEstimator\"",
	      "E", 1, "-E <estimator specification>"));
 
   newVector.addElement(new Option(
	      "\tSwitch on verbose mode and give list of output options.\n"
	      + "\teg: 1,2,11",
	      "V", 1, "-V <option list>"));

    Enumeration enu = super.listOptions();
    while (enu.hasMoreElements()) {
      newVector.addElement(enu.nextElement());
    }

    return newVector.elements();
  }

  /**
   * Parses a given list of options.<p>
   <!-- options-start -->
   <!-- options-leftEnd -->
   *
   * @param options the list of options as an array of strings
   * @exception Exception if an option is not supported
   */
  public void setOptions(String[] options) throws Exception {
    
    String outputRange = Utils.getOption('V', options);
    setVerboseLevels(outputRange);

    boolean doDiscretize = Utils.getFlag('I', options);
    if (doDiscretize) {
      setUseDiscretizer(true);
      //setDiscretizer(new weka.filters.supervised.attribute.Discretize());
    }

    // set estimators options
    // actually uses only the option string to initialize TUBEstimator
    //boolean estimatorSet = false;
    String [] estSpec = null;
    String estName = "weka.estimators.TUBEstimator";
    String estString = Utils.getOption('E', options);
    if (estString.length() != 0) {
      estSpec = Utils.splitOptions(estString);
      if (estSpec.length == 0) {
	throw new IllegalArgumentException("Invalid estimator specification string");
      }
      estName = estSpec[0];
      estSpec[0] = "";
      //estimatorSet = true;
    }
 
    // set discretizing filter and its options
    // if estimator was set, hands the estimator to the filter
    String disString = Utils.getOption('Z', options);
    if (disString.length() != 0) {
 //      if (!doDiscretize) {
// 	throw new Exception("Discretizer can only be set with the use-discretization flag set.");
//       }
    
      String [] disSpec = Utils.splitOptions(disString);
      if (disSpec.length == 0) {
	throw new IllegalArgumentException("Invalid discretizing filter specification string");
      }
      String disName = disSpec[0];
      disSpec[0] = "";
      setDiscretizer((Filter) Utils.forName(Filter.class, disName, disSpec));
          
    } else {
      // set estimator in this class
      setEstimator((Estimator) Utils.forName(Estimator.class, estName, estSpec));
    }

    super.setOptions(options);
  }

  /**
   * Gets the current settings of the classifier.
   *
   * @return an array of strings suitable for passing to setOptions
   */
  public String [] getOptions() {

    Vector result = new Vector();

    // options of the superclass
    String [] superOptions = super.getOptions();
    for (int i = 0; i < superOptions.length; i++) {
      result.add(superOptions[i]);
    }

    // own options
    result.add("-E");
    result.add("" + getEstimatorSpec());

    if (getUseDiscretizer()) {
      result.add("-I");
    }

    result.add("-Z");
    result.add("" + getDiscretizerSpec());

    String verboseLevels = getVerboseLevels();
    if (verboseLevels.length() > 0) {
      result.add("-V");
      result.add("" + verboseLevels);

    }
    
    return (String[])result.toArray(new String[result.size()]);
  }

   /**
   * Gets the estimator specification string.
   *
   * @return the filter string.
   */
  protected String getEstimatorSpec() {
    
    Estimator e = getEstimator();
    if (e == null) return "";
    if (e instanceof OptionHandler) {
      return e.getClass().getName() + " "
	+ Utils.joinOptions(((OptionHandler) e).getOptions());
    }
    return e.getClass().getName();
  }

   /**
   * Gets the discretizing filter specification string.
   *
   * @return the filter string.
   */
  protected String getDiscretizerSpec() {
    
    Filter d = getDiscretizer();
    if (d == null) return "";
    if (d instanceof OptionHandler) {
//       String [] os = ((OptionHandler) d).getOptions();
//       for (int i = 0; i < os.length; i++) {
// 	DBO.pln("getDiscretizerSpec.-joinoptions "+os[i]);
//       }
           return d.getClass().getName() + " "
      	+ Utils.joinOptions(((OptionHandler) d).getOptions());
    }
    return d.getClass().getName();
  }

  /**
   * Returns a description of the classifier.
   *
   * @return a description of the classifier as a string.
   */
  public String toString() {
    
    StringBuffer text = new StringBuffer();

    text.append("Naive Bayes Classifier");
    if (m_Instances == null) {
      text.append(": No model built yet.");
    } else {
      try {
	for (int i = 0; i < m_Distributions[0].length; i++) {
	  text.append("\n\nClass " + m_Instances.classAttribute().value(i) +
		      ": Prior probability = " + Utils.
		      doubleToString(m_ClassDistribution.getProbability(i),
				     4, 2) + "\n\n");
	  Enumeration enumAtts = m_Instances.enumerateAttributes();
	  int attIndex = 0;
	  while (enumAtts.hasMoreElements()) {
	    Attribute attribute = (Attribute) enumAtts.nextElement();
	    text.append(attribute.name() + ":  " 
			+ m_Distributions[attIndex][i]);
	    attIndex++;
	  }
	}
      } catch (Exception ex) {
	text.append(ex.getMessage());
      }
    }

    return text.toString();
  }

  /**
   * Switches the outputs on that are requested from the option V
   * if list is empty switches on the verbose mode only
   * @param list list of integers, all are used for an output type
   */
  public void setVerboseLevels(String list) { 
    dbo.setOutputTypes(list);
  }

  /**
   * Gets the current output type selection
   *
   * @return a string containing a comma separated list of ranges
   */
  public String getVerboseLevels() {
    return dbo.getOutputTypes();
  }
  
  /**
   * Sets the estimator
   *
   * @param estimator the estimator with all options set.
   */
  public void setEstimator(Estimator estimator) {

    m_estimator = estimator;
  }

  /**
   * Gets the estimator used.
   *
   * @return the estimator
   */
  public Estimator getEstimator() {

    return m_estimator;
  }

  /**
   * Sets the discretizing filter
   *
   * @param filter the filter with all options set.
   */
  public void setDiscretizer(Filter filter) {

    m_UseDiscretizer = true;
    m_discretizer = filter;
  }

  /**
   * Gets the discretizing filter used
   *
   * @return the filter
   */
  public Filter getDiscretizer() {

    return m_discretizer;
  }
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String useDiscretizerTipText() {
    return "Use discretization to convert numeric attributes (according to the filters range parameter) to nominal "
      +"ones.";
  }

  /**
   * Get whether discretization is to be used.
   *
   * @return true if supervised discretization is to be used.
   */
  public boolean getUseDiscretizer() {
    
    return m_UseDiscretizer;
  }
  
  /**
   * Set whether discretization is to be used.
   *
   * @param usedisc true if discretization is to be used.
   */
  public void setUseDiscretizer(boolean usedisc) {
    
    m_UseDiscretizer = usedisc;
    if (usedisc) {
      if (m_discretizer == null) {
	m_discretizer = new weka.filters.supervised.attribute.Discretize();
      }
   }
  }

  
  /**
   * Main method for testing this class.
   *
   * @param argv the options
   */
  public static void main(String [] argv) {

    try {
      NaiveBayesParametrized nbe = new NaiveBayesParametrized();
      System.out.println(Evaluation.evaluateModel(nbe, argv));
      //System.out.println("illegalcuts "+nbe.measureIllegalCuts());
    } catch (Exception e) {
      e.printStackTrace();
      System.err.println(e.getMessage());
    }
  }
  
  /**
   * Returns the revision string.
   * 
   * @return		the revision
   */
  public String getRevision() {
    return RevisionUtils.extract("$Revision: 1.00 $");
  }

}



