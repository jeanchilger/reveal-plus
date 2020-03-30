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
 *    CutInfo.java
 *    Copyright (C) 2004 Gabi Schmidberger
 *
 */

package weka.filters.unsupervised.attribute;

import java.io.Serializable;
  
/** 
 * Class representing a Discretization with cutpoints and info about
 * the interval boundaries.<p>
 *
 * @author Gabi Schmidberger (gabi@cs.waikato.ac.nz)
 * @version $Revision: 0.0 $
 */

public class CutInfo implements Serializable{

  /** cutpoints */
  public double [] m_cutPoints = null;

  /** info about the interval boundaries */
  public boolean [] m_cutAndLeft = null;


  /** Constructor */
  public CutInfo(int length) {
    m_cutPoints = new double[length];
    m_cutAndLeft = new boolean[length];
  }

  /** Constructor */
  public CutInfo(double [] cutPoints, boolean [] cutAndLeft) {
    m_cutPoints = cutPoints;
    m_cutAndLeft = cutAndLeft;
  }

  /**
   * Returns the number of cutpoints stored in this object.
   * @return number of cutpoints
   */
  public int numCutPoints() {
    if (m_cutPoints == null) return 0;
    return m_cutPoints.length;
  }

  /**
   * Returns a string representation of the infomation about 
   * the cutpoints.
   * @param b beginning of line
   * @return string representing the object
   */
  public String makeString(String b) {
    StringBuffer text = new StringBuffer(""+b+m_cutPoints.length+" cutpoints");
    
    if (m_cutPoints == null) {
      text.append("\n"+b+"No cutpoints found.\n");
    } else {
      for (int i = 0; i < m_cutPoints.length; i++) {
	text.append(""+b+m_cutPoints[i]+" "+m_cutAndLeft[i]+"\n");
      }
      text.append(""+b+"\n");
    }
    return text.toString();
  }
  
//   public static void main(String [] argv) {

// read cutpoints and cutandleft from the command line.
//     try {
//       cutpoints = new CutPoints();
//     } catch (Exception ex) {
//       ex.printStackTrace();
//       System.out.println(ex.getMessage());
//     }
//   }

}
