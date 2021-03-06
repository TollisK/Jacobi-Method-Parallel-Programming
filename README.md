# Jacobi Method Parallel Programming
Compute the Jacobi Method using multiple threads with MPI and OpenMP 

## Jacobi_serial
Η πρώτη εκδοχή του προγράμματος υλοποιεί σειριακά την μέθοδο Jacobi.
Τέλος έγιναν μερικές βελτιστοποιήσεις για τον γρηγορότερο υπολογισμό της μεθόδου, όπως μετρατροπή της
one_jacobi_iteration() σε inline, η αφαίρεση  μεταβλητών που δεν ήταν απαραίτητες
στην εκτέλεση του προγράμματος και η χρήση της συνάρτησης pow(), έναντι της χρήσης
πολλαπλασιασμού για τον υπολογισμό της δεύτερης δύναμης.
Οπότε βγάζουμε τους εξείς χρόνους για τη σειρικό πρόγραμμα:
![image](https://user-images.githubusercontent.com/75782840/148709797-a3b0f29e-c865-4a8b-bb3d-66aae9ed7cc6.png)

Παρατηρούμε ότι οι παραπάνω χρόνοι είναι μικρότεροι από αυτούς του προγράμματος χωρίς
βελτιστοποιήσεις, που είναι λογικό και αναμενόμενο. Επίσης αξιοσημείοτο είναι ότι ο χρόνος
αυξάνεται γραμμικά με το μέγεθος του προβλήματος. 

## ParallelJacobi
Συμπεριλαμβανομένων των προηγούμενων βελτιστοποιήσεων, στο παράλληλο
πρόγραμμα χρησιμοποιείται η MPI_Cart_coords() για την υλοποίηση MPI παραλληλοποίησης
n\*n διεργασιών.
<br>
Αφότου γίνει η αρχικοποίηση και ο διαχωρισμός των διεργασιών (MPI_Init,
MPI_Comm_size, MPI_Dims_create, MPI_Cart_create, MPI_comm_rank, MPI_Cart_cords) η
διεργασία με rank 0 αποθηκεύει τις τιμές του αρχείου input σε μεταβλητές, τις οποίες κάνει
Broadcast μέσω της συνάρτησης MPI_Bcast(), ώστε να αποφευχθεί η αχρείαστη εκτέλεση 
πολλαπλών scanf(). Ακολούθως, εκτελείται η MPI_Barrier() για τον συγχρονισμό των
διεργασιών και αρχίζει ο χρονομετρητής.
<br>
Μέσω της MPI_Cart_shift() οι διεργασίες γνωρίζουν τους γείτονές τους που
αποθηκεύονται στον πίνακα neighbour_ranks.
Κατόπιν αυτού γίνεται η εκτέλεση του jacobi με MPI.
Καταρχάς, δημιουργούμε datatypes για σειρές και στήλες με
MPI_Type_contiguous(),MPI_Type_vector() και MPI_Type_commit(), ώστε να γίνει
αποδοτικότερα η αποστολή δεδομένων από διεργασία σε διεργασία. Αρχικοποιούνται πίνακες
για γραμμές και στήλες, στους οποίους καταχωρούνται οι αποδεχόνενες τιμές από το
MPI_Irecv() χωρίς την χρήση buffer. Ακολούθως, γίνεται η αποστολή των δεδομένων μέσω
MPI_Isend().
Τα Send/Receive ακολουθεί ο υπολογισμός των datapoints της διεργασίας, εκτώς των
halo points. Για την υλοποίησή του χρησιμοποιέιται το δοσμενο for loop με εξέρεση τις τιμές
που το x ή το y ισούται με 1.
‘Υστερα, υπολογίζονται τα halo points. Πρώτα, εισάγονται στον πίνακα οι τιμές των
εξωτερικών datapoints. Με αυτές γνωστές το μόνο που απομένει είναι ο υπολογισμός των halo
points, που χρησιμοποιέιται ένα διπλό for loop για τις σειρές και ένα ακόμα για τις στήλες.
Γίνεται η μεταφορά u με u_old και το jacobi επαναλλαμβάνεται μέχρι να πληρηθεί η
συνθήκη των επαναλήψεων ή του error.

Οπότε βγάζουμε τους εξείς χρόνους για το παράλληλο:
![image](https://user-images.githubusercontent.com/75782840/148709832-60874c17-de4b-4a90-aaec-922609be73ca.png)
Προσέχουμε ότι ανεξάρτήτως του αριθμού των διεργασιών για το ίδιο μέγεθος προβλήματος
το πρόγραμμα είναι ταχύτερο του γραμμικού, που είναι λογικό καθώς το ίδιο πρόβλημα
χωρίζεται σε μικρότερα υποπροβλήματα που λύνονται ταυτόχρωνα. Όσο προσθέτουμε
διεργασίες τόσο μειώνεται ο χρόνος με εξαίρεση το μικρά προβλήματα όπου ο αριθμός των
εσωτερικών datapoints της κάθε διεργασίας είναι υπερβολικά μικρός για να επηρεάσει την
ταχύτητα του προβλήματος, οπότε δεν βλέπουμε μεγάλη διαφορά.

![image](https://user-images.githubusercontent.com/75782840/148709882-22d8af8b-a1bc-417b-99c9-9f3ed9177583.png)
![image](https://user-images.githubusercontent.com/75782840/148709894-be958d6b-5bda-4244-a2fe-7e4a15df3ede.png)
Παρατηρούμε ότι ανεξαρτήτως το μέγεθος του προβλήματος ο ρυθμός κατά τον οποίο
αυξάνεται η επιτάχυνση μένει σταθερός, με εξαίρεση τα δύο μικρότερα προβλήματα, όπου ο
χρόνος λύσης των υποπροβλημάτων είναι αμελητέος σε σχέση με τον χρόνο ένωσης.
<br>
Για την αποδοτικότητα παρατηρούμε μία αντίθετη τάση, όπου όσο προστίθενται διεργασίες η
αποδοτικότητα μειώνεται. Αυτό είναι αναμενόμενο, καθώς όσο αυξάνονται οι διεργασίες
αυξάνεται και η πιθανότητα μία διεργασία να πάρει σχετικά πολύ, καθυστερώντας τις
διεργασίες που έχουν τελειώσει πιο γρήγορα, κάνοντάς τες να περιμένουν και κατά συνέπεια
μειώνοντας την αποδοτικότητα.

## JacobiMP
Το Jacobi_mp.c είναι γρηγορότερο των προηγούμενων, καθώς –
συμπεριλαμβανομένων των προαναφερθέντων βελτιστοποιήσεων- χρησιμοποιεί threads μέσω OpenMP για
τον διαμοιρασμό του υπολογισμόυ των for loops. Κάθε for loop διαχωρίζεται σε threads ίσα με
τον αριθμό των διεργασιών, όπου οι μεταβλητές που υπολογόζονται εσωτερικά είναι private,
ενώ για το error γίνεται reduction(+:). Όλα τα for loops έχουν schedule(static,8), διότι μετά από
πειραματισμούς το 8 φαίνεται να είναι το καλύτερο νούμερο


