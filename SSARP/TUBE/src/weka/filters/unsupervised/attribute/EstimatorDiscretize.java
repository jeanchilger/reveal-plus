//26.5 pc
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
 *    EstimatorDiscretize.java
 *    Copyright (C) 2004 Gabi Schmidberger
 *
 */

package weka.filters.unsupervised.attribute;

import java.io.FileOutputStream;
import java.io.PrintWriter;
import java.util.Enumeration;
import java.util.Vector;

import weka.core.AdditionalMeasureProducer;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Range;
import weka.core.RevisionUtils;
import weka.core.SparseInstance;
import weka.core.Utils;
import weka.core.Capabilities.Capability;
import weka.core.Debug.DBO;
import weka.estimators.BinningEstimator;
import weka.estimators.BinningUtils;
import weka.estimators.Estimator;
import weka.estimators.TUBEstimator;
import weka.filters.DiscretizingFilter;
import weka.filters.Filter;
import weka.filters.UnsupervisedFilter;
 
/** 
 *
 * A filter that discretizes according to the density optimizing the entropy.<p>
 *
 * @author Gabi Schmidberger (gabi@cs.waikato.ac.nz)
 * @version $Revision: 0.0 $
 */
// list Arraylist, linkedlist
public class EstimatorDiscretize extends PotentialClassIgnorer
  implements DiscretizingFilter, OptionHandler, UnsupervisedFilter,
	     AdditionalMeasureProducer
{

  /**
	 * 
	 */
	private static final long serialVersionUID = 3702352229335114200L;

/** Stores which columns to Discretize */
  protected Range m_DiscretizeCols = new Range();

  /** range of outputtype */
  private Range m_outputTypes = new Range();

  /** Store the minimal attribute values */
  protected double [] m_minValues = null;

  /** Store the maximal attribute values */
  protected double [] m_maxValues = null;

  /** Store the current cutpoints */
  protected double [][] m_cutPoints = null;

  /** Store for each cutpoint if point at cut goes to left bin */
  protected boolean [][] m_cutAndLeft = null;

  /** The default columns to discretize */
  protected String m_DefaultCols;

  /** The seed used for discretization */
  protected int m_seed;

  /** Output binary attributes for discretized attributes. */
  protected boolean m_makeBinary = false;

  /** Split after class values before discretizing. */
  protected boolean m_splitIntoClasses = false;

  /** The estimator set by option (only with L option) */
  protected Estimator m_Estimator = new TUBEstimator();

  /** The estimator used */
  protected Estimator m_actEstimator = null;

  /** The options string for the TUBEstimator */
  String [] m_LLOptions = null;

  /** option list for the estimators */
  protected String [] m_estimatorOptions;
  
//todo
  /** list of bins */
  protected Vector m_bins = new Vector(); 

  /** Constructor - initialises the filter */
  public EstimatorDiscretize() {
    m_outputTypes.setUpper(15);
    m_DefaultCols = "first-last";
    setAttributeIndices("first-last");
  }

  /** Another constructor */
  public EstimatorDiscretize(String cols) {
    m_outputTypes.setUpper(15);
    m_DefaultCols = cols;
    setAttributeIndices(cols);
  }
  
  /** 
   * Returns the Capabilities of this filter.
   *
   * @return            the capabilities of this object
   * @see               Capabilities
   */
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();

    // attributes
    result.enableAllAttributes();
    result.enable(Capability.MISSING_VALUES);
    
    // class
    result.enableAllClasses();
    result.enable(Capability.MISSING_CLASS_VALUES);
    if (!getMakeBinary())
      result.enable(Capability.NO_CLASS);
    
    return result;
  }

  /**
   * Returns an enumeration of the additional measure names.
   *
   * @return an enumeration of the measure names
   */
  public Enumeration enumerateMeasures() {
    
    Vector newVector = new Vector(2);
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
   * Measure function for difference of illegalCuts.
   *
   * @return the tree size
   */
  public double measureAvgCVIllegalCuts() {
    
    if (m_Estimator instanceof BinningEstimator) {
      return ((BinningEstimator)m_actEstimator).getAvgCVIllegalCuts();
    } else {
      return 0.0;
    }
  }

  /**
   * Measure function for illegalCuts.
   *
   * @return the tree size
   */
  public double measureIllegalCuts() {
    
    if (m_Estimator instanceof BinningEstimator) {
      return ((BinningEstimator)m_actEstimator).getNumIllegalCuts();
    } else {
      return 0.0;
    }
  }

  /**
   * Measure function for difference of illegalCuts.
   *
   * @return the tree size
   */
  public double measureDiffIllegalCuts() {
    
    if (m_Estimator instanceof BinningEstimator) {
      return ((BinningEstimator)m_actEstimator).getDiffNumIllegalCuts();
    } else {
      return 0.0;
    }
  }

  /**
   * Returns the tip text for this property
   *
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String attributeIndicesTipText() {
    return "Specify range of attributes to act on."
      + " This is a comma separated list of attribute indices, with"
      + " \"first\" and \"last\" valid values. Specify an inclusive"
      + " range with \"-\". E.g: \"first-3,5,6-10,last\".";
  }

  /**
   * Gets the current range selection
   *
   * @return a string containing a comma separated list of ranges
   */
  public String getAttributeIndices() {

    return m_DiscretizeCols.getRanges();
  }

  /**
   * Sets which attributes are to be Discretized (only numeric
   * attributes among the selection will be Discretized).
   *
   * @param rangeList a string representing the list of attributes. Since
   * the string will typically come from a user, attributes are indexed from
   * 1. <br>
   * eg: first-3,5,6-last
   * @exception IllegalArgumentException if an invalid range list is supplied 
   */
  public void setAttributeIndices(String rangeList) {

    m_DiscretizeCols.setRanges(rangeList);
  }

  /**
   * Returns the tip text for this property
   *
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String makeBinaryTipText() {
    return "Make resulting attributes binary.";
  }

  /**
   * Gets whether binary attributes should be made for discretized ones.
   *
   * @return true if attributes will be binarized
   */
  public boolean getMakeBinary() {
    return m_makeBinary;
  }

  /** 
   * Sets whether binary attributes should be made for discretized ones.
   * @param makeBinary new flag value
   */
  public void setMakeBinary(boolean makeBinary) {
    m_makeBinary = makeBinary;
  }
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String splitIntoClassesTipText() {
    return "Split into classvalues before discretizing.";
  }

  /**
   * Gets whether binary attributes should be made for discretized ones.
   * @return true if attributes will be binarized
   */
  public boolean getSplitIntoClasses() {
    return m_splitIntoClasses;
  }

  /** 
   * Sets whether dataset should be split into sets for each class
   * value before discretizing
   * @param split new flag value
   */
  public void setSplitIntoClasses(boolean split) {
    m_splitIntoClasses = split;
  }

   /**
   * Sets the maximum number of bins
   *
   * @param max the maximum number of splits
   */
  public void setNumBins(int max) {

    // setMaxSplits(max - 1);
  }

  /**
   * Return a string representing the range of output types, that
   * are switched on.
   * @return string representing the range of output
   */
  public String getOutputTypes() { 
    return m_outputTypes.getRanges();
  }

  /**
   * Switches the outputs on that are requested from the option O
   * @param list list of integers, all are used for an output type
   */
  public void setOutputTypes(String list) { 
    if (list.length() == 0) return;
    m_outputTypes.setRanges(list);
    m_outputTypes.setUpper(15);

    /**  4 -> 3  is list of all cutpoints */
  }

  /**
   * Return true if the outputtype is set
   * @param num value that is reserved for a specific outputtype
   * @return return true if the output type is set
   */
  public boolean outputTypeSet(int num) {
    // return false;
    return (m_outputTypes.isInRange(num));
  }
  
  /**
   * Gets the current range selection as range
   * @return a range of attributes that the filter discretizes
   */
  public Range getDiscretizeCols() {

    return m_DiscretizeCols;
  }

 /**
   * Sets the format of the input instances.
   *
   * @param instanceInfo an Instances object containing the input instance
   * structure (any instances contained in the object are ignored - only the
   * structure is required).
   * @return true if the outputFormat may be collected immediately
   * @exception Exception if the input format can't be set successfully
   */
  public boolean setInputFormat(Instances instanceInfo) throws Exception {

    //DBO.pln("setInputFormat");
    if (m_makeBinary && m_IgnoreClass) {
      throw new IllegalArgumentException("Can't ignore class when " +
					 "changing the number of attributes!");
    }

    m_DiscretizeCols.setUpper(instanceInfo.numAttributes() - 1);
    m_cutPoints = null;

    super.setInputFormat(instanceInfo);
    return false;
  }

  /**
   * Gets the flag if decision on split is done by cross validating
   * @return the setting of the cross validatining flag
   */
  public int getSeed() {
    return m_seed;
  }

  /**
   * Sets the value for the random seed
   * @param newFlag the new flag value
   */
  public void setSeed(int seed) {
    m_seed = seed;
  }


  /**
   * Writes the attribute to the outputfile, adding an attribute containing the bin number.
   * @param name of the outputfile
   * @exception if new PrintWriter doesn't work
   */
  public void writeToOutput(String f, int attrIndex, Instances data, double [] cutPoints, boolean [] cutAndLeft, Vector bins) throws Exception{
    PrintWriter output = null;
    Bin bin = null;
    
    if (f.length() != 0) {
      // add attribute indexnumber to filename and extension .arff
      String name = f + attrIndex+".arff";
      output = new PrintWriter(new FileOutputStream(name));
    } else {
      return;
    }
    // make new attribute
    FastVector atts = new FastVector(2);
    // the values themselves
    Attribute att = new Attribute(data.attribute(attrIndex).name());
    atts.addElement(att);
    // the density values
    att = new Attribute("density");
    atts.addElement(att);
    // the discretized value= bin number
    att = setAttributeFormat(data, attrIndex);
    atts.addElement(att);
    Instances newData = new Instances(data.relationName()+"_dis_"+attrIndex,
				      atts, 0);
    
    for (int i = 0; i < data.numInstances(); i++) {
      double [] vals = new double[3];
      vals[0] = data.instance(i).value(attrIndex);
      // no bins or only one
      if (cutPoints == null || cutPoints.length == 0) {
	if (bins != null) {
	  bin = (Bin)bins.elementAt(0);
	  vals[1] =  bin.getDensity();
	}
	vals[2] = 0.0;
      } else {
	int j = cutPoints.length - 1;
	vals[2] = -1.0;
	do {
	  if ((vals[0] > cutPoints[j]) ||
	      ((vals[0] == cutPoints[j]) && !cutAndLeft[j])) {
	    bin = (Bin)bins.elementAt(j+1);
	    vals[1] = bin.getDensity();
	    vals[2] = (double)(j + 1);
	  }
	  j--;
	} while ( vals[2] == -1.0 && j >= 0);
	// instance is in bin 0
	if (vals[2] == -1.0) {
	  bin = (Bin)bins.elementAt(0);
	  vals[1] = bin.getDensity();
	  vals[2] = (double)(0);
	}
      }
      Instance inst = new Instance(data.instance(i).weight(), vals);
      inst.setDataset(newData);

      newData.add(inst);
    }
    output.println(newData.toString());    

    // close output
    if (output != null) {
      output.close();
    }
  }

  public void writeHistogram(String f, Vector bins) throws Exception{
    PrintWriter output = null;
    Bin bin = null;
    StringBuffer text = new StringBuffer("");
    
    if (f.length() != 0) {
      // add attribute indexnumber to filename and extension .arff
      String name = f + "H.arff";
      output = new PrintWriter(new FileOutputStream(name));
    } else {
      return;
    }
    if (bins == null) return;
    if (bins.size() == 0) return;

    // first bin
    bin = (Bin) bins.elementAt(0);
    try {
      text.append("" + bin.getMinValue()+" "+0.0+" ");
      
      for (int i = 0; i < bins.size(); i++) {
	bin = (Bin) bins.elementAt(i);
	text.append(""+bin.getMinValue()+" "+bin.getDensity()+" ");
	text.append(""+bin.getMaxValue()+" "+bin.getDensity()+" ");
      }
      text.append("" + bin.getMaxValue()+" "+0.0+" ");
      // last bin
    } catch (Exception ex) {
      ex.printStackTrace();
      System.out.println(ex.getMessage());
    }
    output.println(text.toString());    
  }

  /**
   * Returns the tip text for this property
   *
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String invertSelectionTipText() {

    return "Set attribute selection mode. If false, only selected"
      + " (numeric) attributes in the range will be discretized; if"
      + " true, only non-selected attributes will be discretized.";
  }

  /**
   * Gets whether the supplied columns are to be removed or kept
   *
   * @return true if the supplied columns will be kept
   */
  public boolean getInvertSelection() {

    return m_DiscretizeCols.getInvert();
  }

  /**
   * Sets whether selected columns should be removed or kept. If true the 
   * selected columns are kept and unselected columns are deleted. If false
   * selected columns are deleted and unselected columns are kept.
   *
   * @param invert the new invert setting
   */
  public void setInvertSelection(boolean invert) {
    m_DiscretizeCols.setInvert(invert);
  }

  /**
   * Sets which attributes are to be Discretized (only numeric
   * attributes among the selection will be Discretized).
   *
   * @param attributes an array containing indexes of attributes to Discretize.
   * Since the array will typically come from a program, attributes are indexed
   * from 0.
   * @exception IllegalArgumentException if an invalid set of ranges
   * is supplied 
   */
  public void setAttributeIndicesArray(int [] attributes) {
    setAttributeIndices(Range.indicesToRangeList(attributes));
  }

  /**
   * Get column range that is to be discretized.
   */
  public Range get_DiscretizeCols() { 
    return m_DiscretizeCols;
  }

  /**
   * Gets an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   */
  public Enumeration listOptions() {

    Vector newVector = new Vector(7);

    newVector.addElement(new Option(
              "\tSpecifies list of columns to Discretize. First"
	      + " and last are valid indexes.\n"
	      + "\t(default: first-last)",
              "R", 1, "-R <col1,col2-col4,...>"));

    newVector.addElement(new Option(
              "\tInvert matching sense of column indexes.",
              "V", 0, "-V"));

    newVector.addElement(new Option(
              "\tOutput binary attributes for discretized attributes.",
              "D", 0, "-D"));


    return newVector.elements();
  }

  /**
   * Parses the options for this object. Valid options are: <p>
   *
   * -B num <br>
   * Specifies the (maximum) number of bins to divide numeric attributes into.
   * Default = 10.<p>
   *
   * -R col1,col2-col4,... <br>
   * Specifies list of columns to Discretize. First
   * and last are valid indexes. (default none) <p>
   *
   * -V <br>
   * Invert matching sense.<p>
   *
   * -D <br>
   * Make binary nominal attributes. <p>
   * 
   * @param options the list of options as an array of strings
   * @exception Exception if an option is not supported
   */
  public void setOptions(String[] options) throws Exception {

    setInvertSelection(Utils.getFlag('V', options));
    
    String convertList = Utils.getOption('R', options);
    if (convertList.length() != 0) {
      setAttributeIndices(convertList);
    } else {
      setAttributeIndices(m_DefaultCols);
    }

    String outputRange = Utils.getOption('O', options);
    setOutputTypes(outputRange);

    setSplitIntoClasses(Utils.getFlag('C', options)); //todo

    setMakeBinary(Utils.getFlag('D', options)); //todo
    
    String estString = Utils.getOption('E', options);
    if (estString.length() != 0) {
    	String [] estSpec = Utils.splitOptions(estString);
    	if (estSpec.length == 0) {
    		throw new IllegalArgumentException("Invalid estimator specification string");
    	}
      //      setLLEstimatorOptions(estSpec);
      String estName = estSpec[0];
      estSpec[0] = "";
      setEstimator((Estimator) Utils.forName(Estimator.class, estName, estSpec));
     }
//     // save options for the estimators
//     m_estimatorOptions = options;
    
//     // see if there are errors    
//     Estimator estimator = new TUBEstimator();
//     ((TUBEstimator)estimator).setOptions(options); 
    
    if (getInputFormat() != null) {
      setInputFormat(getInputFormat());
    }
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String estimatorTipText() {
    return "The estimator to be used.";
  }

  /**
   * Sets the estimator
   *
   * @param estimator the estimator with all options set.
   */
  public void setEstimator(Estimator estimator) {

    m_Estimator = estimator;
    //DBO.pln("setEstimator "+getEstimatorSpec() );
  }

  /**
   * Gets the estimator used.
   *
   * @return the estimator
   */
  public Estimator getEstimator() {

    return m_Estimator;
  }

  /**
   * Gets the classifier specification string, which contains the class name of
   * the classifier and any options to the classifier
   *
   * @return the estimator string.
   */
  protected String getEstimatorSpec() {
    
    Estimator est = getEstimator();
    if (est instanceof OptionHandler) {
      return est.getClass().getName() + " "
	+ Utils.joinOptions(((OptionHandler)est).getOptions());
    }
    return est.getClass().getName();
  }


//    /**
//    * Sets the estimator options used for the TUBEstimator
//    *
//    * @param estimator the estimator with all options set.
//    */
//   public void setLLEstimatorOptions(String [] opt) {

//     m_LLOptions = opt;
//   }

//   /**
//    * Gets the estimator options used for the TUBEstimator.
//    *
//    * @return the options string array
//    */
//   public String []  getLLEstimatorOptions() {

//     return m_LLOptions;
//   }

 /**
   * Gets the current settings of the filter.
   *
   * @return an array of strings suitable for passing to setOptions
   */
  public String [] getOptions() {

    String [] options = new String [20];
    int current = 0;

    if (!getAttributeIndices().equals("")) {
      options[current++] = "-R"; options[current++] = getAttributeIndices();
    }
    if (getMakeBinary()) {
      options[current++] = "-D";
    }
    if (getSplitIntoClasses()) {
      options[current++] = "-C";
    }

    options[current++] = "-E";
    options[current++] = "" + getEstimatorSpec();

    while (current < options.length) {
      options[current++] = "";
    }
    return options;
  }

  /**
   * Input an instance for filtering. 
   *
   * @param instance the input instance
   * @return true if the filtered instance may now be
   * collected with output().
   * @exception IllegalStateException if no input format has been defined.
   */
  public boolean input(Instance instance) {
//  DBO.pln("input");
    if (getInputFormat() == null) {
      throw new IllegalStateException("No input instance format defined");
    }
    if (m_NewBatch) {
      resetQueue();
      m_NewBatch = false;
    }
    
//     if (m_cutPoints == null) {
//       DBO.pln("calculateCutPoints");
//       calculateCutPoints();
//     }
    if (m_cutPoints != null) {
      convertInstance(instance);
      return true;
    }

    bufferInput(instance);
    return false;
  }

  /**
   * Signifies that this batch of input to the filter is finished. If the 
   * filter requires all instances prior to filtering, output() may now 
   * be called to retrieve the filtered instances.
   *
   * @return true if there are instances pending output
   * @exception IllegalStateException if no input structure has been defined
   */
  public boolean batchFinished() {
    if (getInputFormat() == null) {
      throw new IllegalStateException("No input instance format defined");
    }

    if (m_cutPoints == null) {
      calculateCutPoints();
      //DBO.pln("batchFinished "+getEstimator().toString());
      ///todo
      // getcutpoints
      setOutputFormat();
      for(int i = 0; i < getInputFormat().numInstances(); i++) {
	convertInstance(getInputFormat().instance(i));
      }
    } 
   
    flushInput();

    m_NewBatch = true;
    return (numPendingOutput() != 0);
  }

  /**
   * Returns a string describing this filter
   *
   * @return a description of the filter suitable for
   * displaying in the explorer/experimenter gui
   */
  public String globalInfo() {

    return "An instance filter that discretizes a range of numeric"
      + " attributes in the dataset into nominal attributes."
      + " Discretization is by simple binning. Skips the class"
      + " attribute if set.";//todo
  }

  /**
   * Get the minimal value of an attribute
   * @param the index (from 0) of the attribute to get the cut points of
   * @return the minimal value of an attribute
   */
  public double getMinValue(int attributeIndex) {

    if (m_minValues == null) {
      return Double.MIN_VALUE;
    }
    return m_minValues[attributeIndex];
  }

  /**
   * Get the maximal value of an attribute
   * @param the index (from 0) of the attribute to get the cut points of
   * @return the maximal value of an attribute
   */
  public double getMaxValue(int attributeIndex) {

    if (m_maxValues == null) {
      return Double.MAX_VALUE;
    }
    return m_maxValues[attributeIndex];
  }

  /**
   * Get the cut points for an attribute
   * @param the index (from 0) of the attribute to get the cut points of
   * @return an array containing the cutpoints (or null if the
   * attribute requested has been discretized into only one interval.)
   */
  public double [] getCutPoints(int attributeIndex) {

    if (m_cutPoints == null) {
      return null;
    }
    return m_cutPoints[attributeIndex];
  }

  /**
   * Get the info about the bin borders for one attribute.
   * @param index the index of the attribute
   * @return the cutpointinformation of this attribute
   */
  public boolean [] getCutAndLeft(int index) {
    if (m_cutAndLeft == null) {
      return null;
    }
    return m_cutAndLeft[index];
  }
  
  /** Generate the cutpoints for each attribute 
   */
  protected void calculateCutPoints() {
    
    m_minValues = new double [getInputFormat().numAttributes()];
    m_maxValues = new double [getInputFormat().numAttributes()];
    m_cutPoints = new double [getInputFormat().numAttributes()] [];
    m_cutAndLeft = new boolean [getInputFormat().numAttributes()] [];
    
    for(int attrIndex = getInputFormat().numAttributes() - 1; attrIndex >= 0; 
	attrIndex--) {
      if ((m_DiscretizeCols.isInRange(attrIndex)) && 
	  (getInputFormat().attribute(attrIndex).isNumeric()) &&
	  (getInputFormat().classIndex() != attrIndex)) {

	  Instances workData = new Instances(getInputFormat());
	  try {
	    calculateCutPointsUsingEstimator(attrIndex, workData, 
					     getSplitIntoClasses());
	  } catch (Exception ex) {
	    ex.printStackTrace();
	    System.out.println(ex.getMessage());
	  }
	
      }
    }
  }

   /*
   * Class to represent a cutpoint.
   */
  private class CutCombo {
    double cut = Double.NaN;
    boolean left = true;
    int leftOrRight = 0; // > 0 is right, < 0 is left

    /** Constructor
     *@param c cut value
     *@param l border indicator, if true border goes to left
     */ 
    CutCombo(double c, boolean l) {
      cut = c;
      left = l;
      if (left) { leftOrRight--; } else { leftOrRight++; }
    }

    /**
     * add one more
     */
    void addSame(boolean l) {
      if (left) { leftOrRight--; } else { leftOrRight++; }
    }

    /**
     * get the border indicator, if true instance at border goes to left
     */
    boolean getLeft() {
      boolean newLeft = left;
      if (leftOrRight > 0) return false;
      else return true;
    }

  }

  /**
   * Mix a matrix of cut points (of all class values) into a vector of cut points
   * @param cut matrix of cut points
   * @param left matrix of border indicators
   * @return cut info with the sorted vector of cut points and its border indicators
   */
  private CutInfo mixCutPoints(double [][] cut, boolean [][] left) {
    Vector sortedCuts = new Vector();
    CutInfo cInfo = null;
    CutCombo c = null;

    // add first cut in sorted list
    boolean firstFound = false;
    int firsti = 0;
    int firstj = 0;
    for (int i = 0; (!firstFound) && (i < cut.length); i++) {
      for (int j = 0;  (!firstFound) && (j < cut[i].length); j++) {
	c = new CutCombo(cut[i][j], left[i][j]);
	sortedCuts.add(c);
	firsti = i; firstj = j;
	firstFound = true;
	continue;
      }
    }

    if (!firstFound) return cInfo;

    // go through all cuts
    CutCombo nextCut = null;
    for (int i = firsti; i < cut.length; i++) {
      nextCut:
      for (int j = firstj; j < cut[i].length; j++) {
	nextCut = new CutCombo(cut[i][j], left[i][j]);

	// find space in sorted vector of cuts
	for (int vIndex = 0; vIndex < sortedCuts.size(); vIndex++) {
	  CutCombo vCut = (CutCombo)sortedCuts.elementAt(vIndex);

	  if (vCut.cut == nextCut.cut) { 
	    // same cut already here, take the first, take left of the first
	    vCut.addSame(nextCut.left);
	    continue nextCut;
	  }
	  if (vCut.cut > nextCut.cut) { 
	    // found right space
	    sortedCuts.add(vIndex, nextCut);
	    continue nextCut;
	  }
	}
	sortedCuts.add(nextCut);
      }
    }
    double [] newCuts = new double[sortedCuts.size()];
    boolean [] newLefts = new boolean[sortedCuts.size()];
    for (int vIndex = 0; vIndex < sortedCuts.size(); vIndex++) {
      CutCombo vCut = (CutCombo)sortedCuts.elementAt(vIndex);
      newCuts[vIndex] = vCut.cut;
      newLefts[vIndex] = vCut.getLeft();
    }
    cInfo = new CutInfo(newCuts, newLefts);
    return cInfo;
  }

  /**
   * Set cutpoints for a single attribute. 
   *
   * @param attrIndex the index of the attribute to set cutpoints for
   * @param data the data of which one attribute is discretized
   * @param splitIntoClasses true if the data should be split into classes
   * @exception an exception thrown if building astimator doesn't work
   */
  protected void calculateCutPointsUsingEstimator(int attrIndex, Instances data,
						  boolean splitIntoClasses) 
    throws Exception {
    
    CutInfo info = null;
    double min = Double.NaN;
    double max = Double.NaN;
    
    if (!splitIntoClasses) {
      // make a copy of the estimator
      m_actEstimator = Estimator.makeCopy((Estimator)m_Estimator);
      
      // build estimator
      ((Estimator)m_actEstimator).addValues(data, attrIndex);
      
      // getinformation about cut values
      info = ((BinningEstimator)m_actEstimator).getCutInfo();
      
      // get min and max values
      min = ((BinningEstimator)m_actEstimator).getMinValue(); 
      max = ((BinningEstimator)m_actEstimator).getMaxValue();
      
    } else {
      int numClasses = data.numClasses();
      double [][] cutPoints= new double[numClasses][];
      boolean [][] cutAndLeft = new boolean[numClasses][];
      
      for (int cl = 0; cl < numClasses; cl++) {
        
        Estimator actEstimator = Estimator.makeCopy((Estimator)m_Estimator);
        //DBO.pln("attrIndex "+attrIndex+" classIndex "+getInputFormat().classIndex());
        ((Estimator)actEstimator).addValues(data, attrIndex, 
            getInputFormat().classIndex(), cl, 
            min, max);
        
        // getinformation about cut values
        info = ((BinningEstimator)actEstimator).getCutInfo();
        
        // store cut points away
        if (info == null) {
          cutPoints[cl]= null;
          cutAndLeft[cl] = null;
        } else {
          cutPoints[cl]= info.m_cutPoints;
          cutAndLeft[cl] = info.m_cutAndLeft;
        }
        
        // get min and max values
        min = ((BinningEstimator)actEstimator).getMinValue(); 
        max = ((BinningEstimator)actEstimator).getMaxValue();
      }
      
      // output cutpoints 
      info = mixCutPoints(cutPoints, cutAndLeft);         
    }
    
    // store cut points away
    if (info == null) {
      m_cutPoints[attrIndex]= null;
      m_cutAndLeft[attrIndex] = null;
    } else {
      m_cutPoints[attrIndex]= info.m_cutPoints;
      m_cutAndLeft[attrIndex] = info.m_cutAndLeft;
    }
    
    // get min and max values
    m_minValues[attrIndex] = min; 
    m_maxValues[attrIndex] = max;
    // __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ 
    if (outputTypeSet(3)) {
      // output cutpoints
      DBO.pln("#EstimatorDiscretize --4--");
      if ((m_cutPoints[attrIndex] == null) || (m_cutPoints[attrIndex].length == 0)) {
	DBO.pln("\n# no cutpoints found - attribute "+attrIndex); 
      } else {
	DBO.pln("\n#* "+m_cutPoints[attrIndex].length+" cutpoint(s) - attribute "
		 +attrIndex); 
	for (int i = 0; i < m_cutPoints[attrIndex].length; i++) {
	  DBO.p("# "+m_cutPoints[attrIndex][i]+" "); 
	  DBO.pln(""+m_cutAndLeft[attrIndex][i]);
	}
	DBO.pln("# leftEnd");
	CutInfo dbinfo = new CutInfo(m_cutPoints[attrIndex], m_cutAndLeft[attrIndex]);
	Vector bins =  BinningUtils.cutInfoToBins(dbinfo, data, attrIndex, 1.0);
	DBO.pln(BinningUtils.binsToString(bins));
      }
    }
    // ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^
  }


  /**
   * Builds the attributes format for one attribute.
   *@param data the data that is used
   *@param attrIndex 
   */
  protected Attribute setAttributeFormat(Instances data, int attrIndex) {

    Attribute newAtt;
    StringBuffer nameStr = new StringBuffer();
    FastVector attribValues = null;

    if (m_cutPoints == null) {
      return null;
    }
    double [] cutPoints = m_cutPoints[attrIndex];
    boolean [] cutAndLeft = m_cutAndLeft[attrIndex];

    // no cutpoints
    if (m_cutPoints[attrIndex] == null) {
      attribValues = new FastVector(1);
      attribValues.addElement("'All'");
      newAtt = new Attribute("class",
					attribValues);
      return newAtt;
    } else {
      // multivalued nominal attribute
      //
      attribValues = new FastVector();
      for(int j = 0; j <= cutPoints.length; j++) {
	
	nameStr = new StringBuffer();
	
	// the first part of the new attribute name
	if (j == 0) {
	  // the first attribute value
	  nameStr.append("'(-inf-");
	} else {
	  // all but the first attribute value
	  
	  // include the border or not
	  if (cutAndLeft[j - 1]) {
	    nameStr.append("'(");
	  } else {
	    nameStr.append("'[");
	  }
	  nameStr.append(Utils.doubleToString(cutPoints[j - 1], 6) + "-");
	}
	
	// the second part of the new attribute name 
	if (j == cutPoints.length) {
	  // the last attribute value
	  nameStr.append("-inf)'"); 
	} else {
	  // every other value
	  nameStr.append(Utils.doubleToString(cutPoints[j], 6));
	  // include the border or not
	  if (cutAndLeft[j]) {
	    nameStr.append("]'");
	  } else {
	    nameStr.append(")'");
	  }
	}
	attribValues.addElement(nameStr.toString());
      }
      newAtt = new Attribute("class", attribValues);
    }
    return newAtt;
  }
  
  /**
   * Set the output format. Takes the currently defined cutpoints and 
   * m_InputFormat and calls setOutputFormat(Instances) appropriately.
   */
  protected void setOutputFormat() {

    StringBuffer nameStr = new StringBuffer();

    if (m_cutPoints == null) {
      setOutputFormat(null);
      return;
    }
    FastVector attributes = new FastVector(getInputFormat().numAttributes());
    int classIndex = getInputFormat().classIndex();
    for(int i = 0; i < getInputFormat().numAttributes(); i++) {
      /*if ( m_cutPoints[i] != null) {
	for (int j = 0; j <  m_cutPoints[i].length; j++) {
	  DBO.p(" "+m_cutPoints[i][j]);
	} DBO.pln(" m_cutPoints["+i+"].length "+m_cutPoints[i].length);
	}*/
      if ((m_DiscretizeCols.isInRange(i)) 
	  && (getInputFormat().attribute(i).isNumeric())
	  && (getInputFormat().classIndex() != i)) {

	// multivalued nominal attribute
	//
	if (!m_makeBinary) {
	  FastVector attribValues = new FastVector(1);
	  // only one bin
	  if (m_cutPoints[i] == null) {
	    attribValues.addElement("'All'");
	    attributes.addElement(new Attribute(getInputFormat().
						attribute(i).name(),
						attribValues));
	  } else {
	    for(int j = 0; j <= m_cutPoints[i].length; j++) {
	      
	      nameStr = new StringBuffer();
	      
	      // the first part of the new attribute name
	      if (j == 0) {
		// the first attribute value
		nameStr.append("'(-inf-");
	      } else {
		// all but the first attribute value
		
		// include the border or not
		if (m_cutAndLeft[i][j - 1]) {
		  nameStr.append("'(");
		} else {
		  nameStr.append("'[");
		}
		nameStr.append(Utils.doubleToString(m_cutPoints[i][j - 1], 6) + "-");
	      }
	      
	      // the second part of the new attribute name 
	      if (j == m_cutPoints[i].length) {
		// the last attribute value
		nameStr.append("-inf)'"); 
	      } else {
		// every other value
		nameStr.append(Utils.doubleToString(m_cutPoints[i][j], 6));
		// include the border or not
		if (m_cutAndLeft[i][j]) {
		  nameStr.append("]'");
		} else {
		  nameStr.append(")'");
		}
	      }
	    attribValues.addElement(nameStr.toString());
	    //DBO.pln("element "+nameStr.toString());
	    }
	    attributes.addElement(new Attribute(getInputFormat().
						attribute(i).name(),
						attribValues));
	  }
	} else {
	  
	  // one valued nominal attribute
	  //
	  if (m_cutPoints[i] == null) {
	    FastVector attribValues = new FastVector(1);
	    attribValues.addElement("'All'");
	    attributes.addElement(new Attribute(getInputFormat().
						attribute(i).name(),
						attribValues));
	  } else {
	    
	    // binary nominal attribute
	    //
	    if (i < getInputFormat().classIndex()) {
	      classIndex += m_cutPoints[i].length - 1;
	    }
	    for(int j = 0; j < m_cutPoints[i].length; j++) {
	      nameStr = new StringBuffer();
	      StringBuffer nameStr2 = new StringBuffer();
	      FastVector attribValues = new FastVector(2);
	      nameStr.append("'(-inf-" + Utils.doubleToString(m_cutPoints[i][j], 6));
	      // include the border or not
	      if (m_cutAndLeft[i][j]) {
		nameStr.append("]'");
		nameStr2.append("'(");
	      } else {
		nameStr.append(")'");
		nameStr2.append("'[");
	      }
	      
	      attribValues.addElement(nameStr.toString());
	      
	      nameStr2.append(Utils.doubleToString(m_cutPoints[i][j], 6) + "-inf)'");
	      attribValues.addElement(nameStr2.toString());
	      
	      attributes.addElement(new Attribute(getInputFormat().
						  attribute(i).name(),
						  attribValues));
	    }
	  }
	}
      } else {
	attributes.addElement(getInputFormat().attribute(i).copy());
      }
    }
    Instances outputFormat = 
      new Instances(getInputFormat().relationName(), attributes, 0);
    //xDBO.pln("outputFormat\n"+outputFormat);
    outputFormat.setClassIndex(classIndex);
    setOutputFormat(outputFormat);
  }

  /**
   * Convert a single instance over. The converted instance is added to 
   * the leftEnd of the output queue.
   *
   * @param instance the instance to convert
   */
  protected void convertInstance(Instance instance) {

    int index = 0;
    double [] vals = new double [outputFormatPeek().numAttributes()];
    // Copy and convert the values
    for(int i = 0; i < getInputFormat().numAttributes(); i++) {

      // this attribute is to be discretized
      if (m_DiscretizeCols.isInRange(i) && 
	  getInputFormat().attribute(i).isNumeric() &&
	  (getInputFormat().classIndex() != i)) {
	// numeric attribute but not class value
	int j;
	double currentVal = instance.value(i);
	// no cut points found
	if (m_cutPoints[i] == null) {
	  if (instance.isMissing(i)) {
	    vals[index] = Instance.missingValue();
	  } else {
	    vals[index] = 0;
	  }
	  index++;
	} else {
	  // make multivalue nominal attribute
	  if (!m_makeBinary) {
	    if (instance.isMissing(i)) {
	      vals[index] = Instance.missingValue();
	    } else {
	      for (j = 0; j < m_cutPoints[i].length; j++) {
		if (currentVal <= m_cutPoints[i][j]) {
		  break;
		}
	      }
	      //DBO.p(""+currentVal+" i "+i +" j "+j+" cut= ");
	      //DBO.pln(""+ m_cutPoints[i][j]);
	      vals[index] = j;
	      if (j < m_cutPoints[i].length) {
		if ((currentVal == m_cutPoints[i][j]) && (!m_cutAndLeft[i][j])) {
		  vals[index] = j + 1;
		}
	      }
	    }
	    index++;
	  // make binary nominal attribute
	  } else {
	    for (j = 0; j < m_cutPoints[i].length; j++) {
	      // missing value
	      if (instance.isMissing(i)) {
                vals[index] = Instance.missingValue();
	      } else if ((currentVal == m_cutPoints[i][j]) && (!m_cutAndLeft[i][j])) {
		// instance on the cut point goes into the right bin
                vals[index] = 1;
	      } else if (currentVal <= m_cutPoints[i][j]) {
		// smaller than cut value
                vals[index] = 0;
	      } else {
		// larger than cut value
                vals[index] = 1;
	      }
	      index++;
	    }
	  }
	}
      } else {
	// no change to attribute value
        vals[index] = instance.value(i);
	index++;
      }
    }
    
    Instance inst = null;
    if (instance instanceof SparseInstance) {
      inst = new SparseInstance(instance.weight(), vals);
    } else {
      inst = new Instance(instance.weight(), vals);
    }
    //DBO.pln("instance "+instance);
    //copyStringValues(inst, false, instance.dataset(), getInputStringIndex(),
    //                 getOutputFormat(), getOutputStringIndex());
    inst.setDataset(getOutputFormat());
    push(inst);
  }


  /**
   * Main method for testing this class.
   *
   * @param argv should contain arguments to the filter: use -h for help
   */
  public static void main(String [] argv) {

    try {
      if (Utils.getFlag('b', argv)) {
 	Filter.batchFilterFile(new EstimatorDiscretize(), argv);
      } else {
	Filter.filterFile(new EstimatorDiscretize(), argv);
      }
    } catch (Exception ex) {
      ex.printStackTrace();
      System.out.println(ex.getMessage());
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
