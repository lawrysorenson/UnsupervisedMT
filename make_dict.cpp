#include <iostream>
#include <algorithm>
#include <vector>
#include <fstream>
#include <cmath>

using namespace std;

void parse_file(string name, vector<string>& words, vector<vector<float>>& emb, int& size)
{

    ifstream file;
    file.open(name);

    int wordCount;
    file >> wordCount >> size;

    emb.resize(wordCount);
    words.resize(wordCount);
    
    for (int i=0;i<wordCount;++i)
    {
        if (i%100==0) cout << i << '/' << wordCount << '\r';
        string word;
        file >> word;
        words[i] = word;

        vector<float>& e = emb[i];
        e.resize(size);

        double mag = 0;
        for (int j=0;j<size;++j)
        {
            float sub;
            file >> sub;
            e[j] = sub;
            mag += sub*sub;
        }
        mag = sqrt(mag);
        for (int j=0;j<size;++j) e[j] /= mag;
    }

    file.close();
}

struct elem{
    int ind;
    float dist;
    elem(int i, float d): ind(i), dist(d)
    {
    }
    bool operator<(const elem & o) const
    {
        return dist < o.dist;
    }
};

int main()
{
    string corename = "vectors";
    string l1 = "fa";
    string l2 = "en";

    int size;
    vector<string> words1;
    vector<vector<float>> emb1;
    parse_file("data/fasttext/" + corename + "-"+l1+".txt", words1, emb1, size);

    cout << "POINT A" << endl;

    int size2;
    vector<string> words2;
    vector<vector<float>> emb2;
    parse_file("data/fasttext/" + corename + "-"+l2+".txt", words2, emb2, size2);
    cout << "POINT B" << endl;

    if (size!=size2) exit(-1);

    for (int i=0;i<words1.size();++i)
    {
        struct elem best(0, 0);
        for (int j=0;j<words2.size();++j)
        {
            cout << "j/" << words2.size() << "\r";
            float dot = 0;
            for (int k=0;k<size;++k) dot += emb1[i][k] * emb2[j][k];
            struct elem test(j, dot);
            if (best < test) best = test;
        }
        cout << words1[i] << ' ' << words2[best.ind] << ' ' << best.dist << endl;
        if (i==100) break;
    }

    //cout << words1.size() << endl;

    //for (string s : words1) cout << s << endl;


    return 0;
}