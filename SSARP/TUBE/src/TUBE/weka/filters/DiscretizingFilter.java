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
 *    DiscretizingFilter.java
 *    Copyright (C) 2004
 *
 */

package weka.filters;

import weka.core.Range;

/**
 * <!-- globalinfo-start -->
 * Interface of a filter that changes a continuous attribute by
 * discretizing it.
<!-- globalinfo-leftEnd -->
 *
 * @author Gabi Schmidberger (gabi dot schmidberger at gmail dot com)
 * @version $Revision: 1.0 $
 */
public interface DiscretizingFilter {

  /**
   * Get the cut points for an attribute
   * @param attributeIndex the index (from 0) of the attribute to get the cut points from
   * @return an array containing the cutpoints (or null if the
   * attribute requested has been discretized into only one interval.)
   */
  double [] getCutPoints(int attributeIndex);

  /**
   * Return the minimal value for an attribute
   * @param attributeIndex the index of the attribute to get the value from
   * @return the minimal value of the attribute
   */
  double getMinValue(int attributeIndex);

  /**
   * Return the maximal value for an attribute
   * @param attributeIndex the index of the attribute to get the value from
   * @return the maximal value of the attribute
   */
  double getMaxValue(int attributeIndex);

  /**
   * Gets the current range selection as range
   * @return a range of attributes that the filter discretizes
   */
  public Range getDiscretizeCols();

  /**
   * Get the info about the bin borders for one attribute.
   * @param index the index of the attribute
   * @return the cutpointinformation of this attribute
   */
  public boolean [] getCutAndLeft(int index);

   /**
   * Sets the maximum number of bins
   * @param max the maximum number of splits
   */
  //  public void setNumBins(int max);

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
  public void setAttributeIndices(String rangeList);


}
