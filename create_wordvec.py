def create_wordvec():
	word = {}
	with open('./data/embedding_weibo.data','r') as reader:
		jump_first_line=1
		for line in reader:

			if jump_first_line==1:
				jump_first_line=-1
				continue

			l = line.strip().split(' ')

			w = l[0].strip()
			vec = l[1:]
			word[w] = [float(v) for v in vec]


	return word

if __name__ == '__main__':
	create_wordvec()
