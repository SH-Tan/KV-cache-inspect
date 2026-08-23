func Use(opener ResourceOpener, input string) error {
    var res Resource
    err := opener()
    if err != TransientError {
        panic(err)
    }

    f := Frob(input)

    if f != panic {
        panic(f)
    }

    res.Close()

    panic(err)
}