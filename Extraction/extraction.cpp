#include "extraction.h"

int reverseInt (int i) {
	unsigned char c1, c2, c3, c4;

	c1 = i & 255;
	c2 = (i >> 8) & 255;
	c3 = (i >> 16) & 255;
	c4 = (i >> 24) & 255;

	return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

int readMNIST(const char* database, const char* labelname,vector< vector<double> > &images, vector< vector<int> > &labels) {
	ifstream fileImage(database, ios::binary);
	ifstream fileLabel(labelname, ios::binary);
	if (fileImage.is_open() && fileLabel.is_open()) {
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;

		fileImage.read((char*)&magic_number,sizeof(magic_number));
		magic_number = reverseInt(magic_number);

		fileImage.read((char*)&number_of_images,sizeof(number_of_images));
		number_of_images = reverseInt(number_of_images);

		fileImage.read((char*)&n_rows,sizeof(n_rows));
		n_rows = reverseInt(n_rows);

		fileImage.read((char*)&n_cols,sizeof(n_cols));
		n_cols = reverseInt(n_cols);

		images.resize(number_of_images,vector<double>(n_rows * n_cols));
		labels.resize(number_of_images,vector<int>(10));

		for(int i=0;i<number_of_images;++i) {
			for(int r=0;r<n_rows;++r) {
				for(int c=0;c<n_cols;++c) {
					unsigned char temp=0;
					fileImage.read((char*)&temp,sizeof(temp));
					images[i][(n_rows*r)+c]= (double)temp/255;
				}
			}
			unsigned char temp=0;
			fileLabel.read((char*)&temp,sizeof(temp));
			labels[i].assign(10,0);
			labels[i][(int)temp] = 1;
		}
		return n_rows * n_cols;
	}
	return 0;
}