package lda

import (
	"encoding/csv"
	"log"
	"math/rand"
	"os"
	"strconv"
	"strings"
)

type LDA struct {
	alpha float64
	beta  float64
}

func getNdw(corpusFile string) [][]int64 {
	var row [3]int
	var numstr string
	f, err := os.Open(corpusFile)
	if err != nil {
		log.Fatal(err)
	}
	lines, err := csv.NewReader(f).ReadAll()
	if err != nil {
		log.Fatal(err)
	}
	Nd := make([][]int64, 0, 0)
	for i, line := range lines {
		s := strings.Split(line, " ")
		if len(s) != 3 {
			log.Fatal("wrong input format")
		}
		for j, numstr := range s {
			num, err := strconv.Atoi(numstr)
			if err != nil {
				log.Fatal(err)
			}
			row[j] = num
		}
		Nd = append(sli, row)

	}
	return Nd
}

func getWords(vocab_file string) []string {
	f, err := os.Open(vocab_file)
	if err != nil {
		log.Fatal(err)
	}
	lines, err := csv.NewReader(f).ReadAll()
	if err != nil {
		log.Fatal(err)
	}
	words := make([]string)
	for _, line := range lines {
		words = append(words, line.Text())
	}
	return words
}

func getWord2int(words []string) (map[string]int64, map[int64]string) {
	var ind int64
	word2int := make(map[string]int64)
	int2word := make(map[int64]string)
	for _, w := range words {
		word2int[w] = ind
		int2word[ind] = w
		ind++
	}
	return word2int, int2word
}

func runLDA(corpus_file string, vocab_file string, T int64, alpha float64,
	beta float64, gibbsIter int, optiLag int) {

	Ndw := getNdw(corpus_file)
	words := getWords(vocab_file)
	word2int, int2word := getWord2int(words)
	w, d := get_wd(Ndw)
	N := len(w)
	D := MaxInt64Slice(d) + 1
	W := len(words)
	z := make([]int64, N, N)
	Nwt := make([][]int64, W, W) //  #np.zeros((W,T), dtype = np.int64)
	Ndt := make([][]int64, D, D) //np.zeros((D,T), dtype = np.int64)

	// Initial values
	for i, j := range rand.Perm(N) {
		z[i] = int64(j)
	}
	for i := 0; i < N; i++ {
		Nwt[w[i]][z[i]]++
		Ndt[d[i]][z[i]]++
	}
	iterateGibbs(z, w, d, Nwt, Ndt, alpha, beta, gibbsIter, optiLag)
}

func getNd(Ndx [][]int64) []int64 {
	D := len(Ndx)
	var rowSum int64
	Nd := make([]int64, D, D)
	for d, row := range Ndx {
		rowSum = 0
		for _, elem := range row {
			rowSum += elem
		}
		Nd[d] = rowSum
	}
	return Nd
}
func getNt(Nwt [][]int64) []int64 {
	T := len(Nwt[0])
	Nt := make([]int64, T, T)
	for _, row := range Nwt {
		for t, nwt := range row {
			Nt[t] += nwt
		}
	}
	return Nt
}

func get_wd(Ndw [][]int64) ([]int64, []int64) {
	var N int64
	for _, row := range Ndw {
		N += row[2]
	}
	w := make([]int64, N, N)
	d := make([]int64, N, N)
	var count int64
	for j := 0; j < len(Ndw); j++ {
		for i := count; i < count+Ndw[i][2]; i++ {
			w[i] = Ndw[j][1]
			d[i] = Ndw[j][0]
		}
		count += Ndw[j][2]
	}
	return w, d
}
