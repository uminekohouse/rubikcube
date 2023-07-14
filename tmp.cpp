#include <bits/stdc++.h>
#define rep(i, n, m) for (int i = (n); i < (m); ++i)
using namespace std;
using ll = long long;
using Graph = vector<vector<int>>;

char func(char c) {
  char res;
  if (c == 'R')
    res = '1';
  if (c == 'O')
    res = '2';
  if (c == 'G')
    res = '3';
  if (c == 'W')
    res = '4';
  if (c == 'B')
    res = '5';
  if (c == 'Y')
    res = '6';
  return res;
}

int main() {
  vector<string> cube(9);
  for (int i = 0; i < 9; ++i)
    cin >> cube[i];
  string cube_string;
  rep(i, 0, 3) rep(j, 3, 6) cube_string += cube[i][j];
  rep(i, 3, 6) rep(j, 6, 9) cube_string += cube[i][j];
  rep(i, 3, 6) rep(j, 3, 6) cube_string += cube[i][j];
  rep(i, 6, 9) rep(j, 3, 6) cube_string += cube[i][j];
  rep(i, 3, 6) rep(j, 0, 3) cube_string += cube[i][j];
  rep(i, 3, 6) rep(j, 9, 12) cube_string += cube[i][j];
  string res;
  for (auto c : cube_string)
    res += func(c);
  cout << cube_string << endl;
  cout << res << endl;
  cout << res.size() << endl;
}
