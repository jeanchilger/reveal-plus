/**
 * Read TAR file in C++
 * Example code
 *
 * (C) Uli Köhler 2013
 * Licensed under CC-By 3.0 Germany: http://creativecommons.org/licenses/by/3.0/de/legalcode
 *
 * Compile like this:
 *   g++ -o cpptar cpptar.cpp -lboost_iostreams -lz -lbz2
 */
#include <cstdlib>
#include <cassert>
#include <cstdio>
#include <fstream>
#include <cmath>
#include <iostream>
#include <map>
//#include <cstdint>
typedef unsigned long long int uint64_t;
#include <cstring>

#include "porter.h"

using namespace std;
map<string,long> df;
long ndocs;

#define ASCII_TO_NUMBER(num) ((num)-48) //Converts an ascii digit to the corresponding number (assuming it is an ASCII digit)

/**
 * Decode a TAR octal number.
 * Ignores everything after the first NUL or space character.
 * @param data A pointer to a size-byte-long octal-encoded
 * @param size The size of the field pointer to by the data pointer
 * @return
 */
static uint64_t decodeTarOctal(char* data, size_t size = 12) {
    unsigned char* currentPtr = (unsigned char*) data + size;
    uint64_t sum = 0;
    uint64_t currentMultiplier = 1;
    //Skip everything after the last NUL/space character
    //In some TAR archives the size field has non-trailing NULs/spaces, so this is neccessary
    unsigned char* checkPtr = currentPtr; //This is used to check where the last NUL/space char is
    for (; checkPtr >= (unsigned char*) data; checkPtr--) {
        if ((*checkPtr) == 0 || (*checkPtr) == ' ') {
            currentPtr = checkPtr - 1;
        }
    }
    for (; currentPtr >= (unsigned char*) data; currentPtr--) {
        sum += ASCII_TO_NUMBER(*currentPtr) * currentMultiplier;
        currentMultiplier *= 8;
    }
    return sum;
}

struct TARFileHeader {
    char filename[100]; //NUL-terminated
    char mode[8];
    char uid[8];
    char gid[8];
    char fileSize[12];
    char lastModification[12];
    char checksum[8];
    char typeFlag; //Also called link indicator for none-UStar format
    char linkedFileName[100];
    //USTar-specific fields -- NUL-filled in non-USTAR version
    char ustarIndicator[6]; //"ustar" -- 6th character might be NUL but results show it doesn't have to
    char ustarVersion[2]; //00
    char ownerUserName[32];
    char ownerGroupName[32];
    char deviceMajorNumber[8];
    char deviceMinorNumber[8];
    char filenamePrefix[155];
    char padding[12]; //Nothing of interest, but relevant for checksum

    /**
     * @return true if and only if
     */
    bool isUSTAR() {
        return (memcmp("ustar", ustarIndicator, 5) == 0);
    }

    /**
     * @return The filesize in bytes
     */
    size_t getFileSize() {
        return decodeTarOctal(fileSize);
    }

    /**
     * Return true if and only if the header checksum is correct
     * @return
     */
    bool checkChecksum() {
        //We need to set the checksum to zer
        char originalChecksum[8];
        memcpy(originalChecksum, checksum, 8);
        memset(checksum, ' ', 8);
        //Calculate the checksum -- both signed and unsigned
        int64_t unsignedSum = 0;
        int64_t signedSum = 0;
        for (int i = 0; i < sizeof (TARFileHeader); i++) {
            unsignedSum += ((unsigned char*) this)[i];
            signedSum += ((signed char*) this)[i];
        }
        //Copy back the checksum
        memcpy(checksum, originalChecksum, 8);
        //Decode the original checksum
        uint64_t referenceChecksum = decodeTarOctal(originalChecksum);
        return (referenceChecksum == unsignedSum || referenceChecksum == signedSum);
    }
};

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <TAR archive>" << endl;
        return 1;
    }
    std :: ios_base :: sync_with_stdio ( false );

    ifstream fin(argv[1], ios_base::in | ios_base::binary);
    //gvc filtering_istream in;
    //Depending on the compression format, select the correct decompressor
    string filename(argv[1]);
    //if (boost::algorithm::iends_with(filename, ".gz")) {
        //in.push(gzip_decompressor());
    //} else if (boost::algorithm::iends_with(filename, ".bz2")) {
        //in.push(bzip2_decompressor());
    //} else if (boost::algorithm::iends_with(filename, ".tar")) {
        //No decompression filter needed
    //} else {
    if (0) {
        cerr << "Unknown file suffix: " << filename << endl;
        return 1;
    }
    //gvc in.push(fin);
    //Initialize a zero-filled block we can compare against (zero-filled header block --> end of TAR archive)
    char zeroBlock[512];
    memset(zeroBlock, 0, 512);
    //Start reading
    bool nextEntryHasLongName = false;
    while (! fin.eof()) { //Stop if end of file has been reached or any error occured
        TARFileHeader currentFileHeader;
        //Read the file header.
        fin.read((char*) &currentFileHeader, 512);
    //When a block with zeroes-only is found, the TAR archive ends here
    if(memcmp(&currentFileHeader, zeroBlock, 512) == 0) {
        cerr << "ERR Found TAR end\n";
        break;
    }
    //Uncomment this to check all header checksums
    //There seem to be TARs on the internet which include single headers that do not match the checksum even if most headers do.
    //This might indicate a code error.
    //assert(currentFileHeader.checkChecksum());

        //Uncomment this to check for USTAR if you need USTAR features
        //assert(currentFileHeader.isUSTAR());

        //Convert the filename to a std::string to make handling easier
    //Filenames of length 100+ need special handling
    // (only USTAR supports 101+-character filenames, but in non-USTAR archives the prefix is 0 and therefore ignored)
        string filename(currentFileHeader.filename, min((size_t)100, strlen(currentFileHeader.filename)));
    //---Remove the next block if you don't want to support long filenames---
    size_t prefixLength = strlen(currentFileHeader.filenamePrefix);
    if(prefixLength > 0) { //If there is a filename prefix, add it to the string. See `man ustar`LON
        filename = string(currentFileHeader.filenamePrefix, min((size_t)155, prefixLength)) + "/" + filename; //min limit: Not needed by spec, but we want to be safe
    }
        //Ignore directories, only handle normal files (symlinks are currently ignored completely and might cause errors)
        if (currentFileHeader.typeFlag == '0' || currentFileHeader.typeFlag == 0) { //Normal file
        //Handle GNU TAR long filenames -- the current block contains the filename only whilst the next block contains metadata
        if(nextEntryHasLongName) {
        //Set the filename from the current header
        filename = string(currentFileHeader.filename);
        //The next header contains the metadata, so replace the header before reading the metadata
        fin.read((char*) &currentFileHeader, 512);
        //Reset the long name flag
        nextEntryHasLongName = false;
        }
        //Now the metadata in the current file header is valie -- we can read the values.
            size_t size = currentFileHeader.getFileSize();
            //Log that we found a file
            cerr << "ERR Found file '" << filename << "' (" << size << " bytes)\n";
            //Read the file into memory
            //  This won't work for very large files -- use streaming methods there!
            char* fileData = new char[size + 1]; //+1: Place a terminal NUL to allow interpreting the file as cstring (you can remove this if unused)
            fileData[size] = 0;
            const char *filesuf = strrchr(filename.c_str(),'/');
            if (!filesuf++) filesuf = filename.c_str();
            ndocs++;
            fin.read(fileData, size);
            {
                map<string,long> tf;
//0000000: fffe 2600
                int mysize = size < 30000 ? size : 30000;
                //cerr << "reading" << (0+fileData[0]) << "\n";
                int i,j;
                if (0&&fileData[0] == -1) {
                   cerr << "utf" << "\n";
                   mysize = size > 60000 ? 30000 : size/2;
                   for (i=0; i<mysize ;i++) {
                      fileData[i] = fileData[2*i];
                      //cerr << "filedat " << fileData[i] << "\n";
                   }
                }
   for (i=0;i+3<mysize;i++){
      char bb[9];
      int k;
      for (j=k=0;j<4;j++){ 
         int c = (unsigned char) fileData[i+j];
         if (c < 9) {
            bb[k++] = '#';
            bb[k++] = '0'+c;
         } else if (c >= 9 && c <= 0xd) bb[k++] = c-9+1;
         else if (c == ' ') bb[k++]=6;
         else if (c == '#') bb[k++]=7;
         else bb[k++]=c;
      }
      bb[k] = 0;
      long x = tf[bb]++;
      if (!x) df[bb]++;
   }
      
                for (i=0;0&&i<mysize;i=j) {
                   int skip = 0;
                   for (;i<mysize && !isalnum(fileData[i]);i++);
                   for (j=i;isalnum(fileData[j]);j++) 
                      if (isdigit(fileData[j])) skip=1;
                      else fileData[j] = tolower(fileData[j]);
                   fileData[j]=0;
                   //cout << fileData+i << ' ';
                   if (!skip) {
                      fileData[stem(fileData,i,j-1)+1]=0;
                      if (fileData[i] && fileData[i+1]) {
                         //cout << fileData+i;
                         long x = tf[fileData+i]++;
                         if (!x) df[fileData+i]++;
                         //cout << ' ' << x;
                      }
                   }
                   //cout << '\n';
                }
                for( map<string,long>::const_iterator it = tf.begin(); it != tf.end(); ++it ) {
                   string key = it->first;
                   long value = it->second;
                   cout << filesuf << ' ' << value << ' ' << key << '\n';
                }
            }
            //-------Place code to handle the file content here---------
            delete[] fileData;
            //In the tar archive, entire 512-byte-blocks are used for each file
            //Therefore we now have to skip the padded bytes.
            size_t paddingBytes = (512 - (size % 512)) % 512; //How long the padding to 512 bytes needs to be
            //Simply ignore the padding
            fin.ignore(paddingBytes);
    //----Remove the else if and else branches if you want to handle normal files only---
        } else if (currentFileHeader.typeFlag == '5') { //A directory
        //Currently long directory names are not handled correctly
            cerr << "ERR Found directory '" << filename << "'\n";
        } else if(currentFileHeader.typeFlag == 'L') {
        nextEntryHasLongName = true;
    } else {
        //Neither normal file nor directory (symlink etc.) -- currently ignored silently
        cerr << "ERR Found unhandled TAR Entry type " << currentFileHeader.typeFlag << "\n";
    }
    }
    //Cleanup
    fin.close();
    ofstream fout;
    fout.open("df");
    for( map<string,long>::const_iterator it = df.begin(); it != df.end(); ++it ) {
       string key = it->first;
       long value = it->second;
       if (value > 1) fout << value << ' ' << key << '\n';
    }
    fout.close();
    fout.open("N");
    fout << ndocs << '\n';
    fout.close();
}
