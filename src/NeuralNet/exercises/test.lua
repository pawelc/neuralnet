


foo = overloaded()

function foo.number(n)
  print("num: "..n)
end

function foo.string(s)
  print("string: "..s)
end

foo(5)
foo("paw")