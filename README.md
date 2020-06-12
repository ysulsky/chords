# Guitar Trainer.

Justin from http://justinguitar.com suggests a helpful exercise to help you with chord changes: pick two chords and switch between them for as many times as you can within a minute.

I found it difficult to count, so I wrote this script to listen and count for me. There's probably a better way to do this, but I don't know any music theory. Instead, you create a training set of the chords you're interested in, plus any background noise you'd like to ignore:

```bash
python3 chords.py --num_chords=2 --duration=10 --mode=collect
python3 chords.py --num_chords=2 --mode=fit
```

This will collect and fit three datasets: a background noise dataset (you should play some other notes/chords on your guitar to feed it negatives), and two chords, ten seconds each. Then test your switching:

```bash
python3 chords.py --duration=60 --mode=exercise
```

The output will be a countdown:
```
5 4 3 2 1 Go!
```

And whenever you switch chords, it will print so on the screen, followed by a counter at the end of the duration. For instance (with my slow switching):

```
None -> 1
1 -> 2
2 -> 1
1 -> 2
2 -> 1
1 -> 2
2 -> 1
1 -> 2
2 -> 1
1 -> 2
2 -> 1
1 -> 2
2 -> 1
1 -> 2
2 -> 1
1 -> 2
2 -> 1
1 -> 2
2 -> 1
1 -> 2
2 -> 1
1 -> 2
2 -> 1
1 -> 2
Total switches: 24
```
